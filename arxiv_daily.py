from llm import *
from util.request import get_yesterday_arxiv_papers
from util.construct_email import *
from util.history import PaperHistory
from tqdm import tqdm
import json
import os
from datetime import datetime, timezone
import time
import random
import smtplib
import re
import requests
from email.header import Header
from email.utils import parseaddr, formataddr
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


class ArxivDaily:
    def __init__(
        self,
        categories: list[str],
        max_entries: int,
        max_paper_num: int,
        provider: str,
        model: str,
        base_url: None,
        api_key: None,
        description: str,
        num_workers: int,
        temperature: float,
        save_dir: None,
        include_seen: bool = False,
        db_path: str = "data/papers.db",
        report_dir: str | None = None,
    ):
        self.model_name = model
        self.base_url = base_url
        self.api_key = api_key
        self.max_paper_num = max_paper_num
        self.save_dir = save_dir
        self.report_dir = report_dir
        self.include_seen = include_seen
        self.history = PaperHistory(db_path)
        self.num_workers = num_workers
        self.temperature = temperature
        self.run_datetime = datetime.now(timezone.utc)
        self.run_date = self.run_datetime.strftime("%Y-%m-%d")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.cache_dir = os.path.join(base_dir, save_dir, self.run_date,"json")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.papers = {}
        for category in categories:
            self.papers[category] = get_yesterday_arxiv_papers(category, max_entries)
            print(
                "{} papers on arXiv for {} are fetched.".format(
                    len(self.papers[category]), category
                )
            )
            # avoid being blocked
            sleep_time = random.randint(5, 15)
            time.sleep(sleep_time)

        provider = provider.lower()
        if provider == "ollama":
            self.model = Ollama(model)
        elif provider == "openai" or provider == "siliconflow" or provider == "deepseek":
            self.model = GPT(model, base_url, api_key)
        else:
            assert False, "Model not supported."
        print(
            "Model initialized successfully. Using {} provided by {}.".format(
                model, provider
            )
        )

        self.description = description
        self.digest_prompt_template = self.load_digest_prompt()
        self.lock = threading.Lock()  # 添加线程锁

    def load_digest_prompt(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        prompt_path = os.path.join(base_dir, "prompts", "quantum_bilingual_digest.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    def clean_model_response(self, raw_text: str) -> str:
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            if "\n" in cleaned:
                first_line, rest = cleaned.split("\n", 1)
                if first_line.strip().lower() in ("json", "html"):
                    cleaned = rest
                else:
                    cleaned = first_line + "\n" + rest
        return cleaned.strip()

    def get_response(self, title, abstract):
        prompt = self.digest_prompt_template.format(
            description=self.description,
            title=title,
            abstract=abstract,
        )

        response = self.model.inference(prompt, temperature=self.temperature)
        return response

    def build_fallback_result(self, paper, relevance_score=0):
        arxiv_url = paper.get("abstract_url") or f"https://arxiv.org/abs/{paper['arXiv_id']}"
        return {
            "title": paper["title"],
            "arXiv_id": paper["arXiv_id"],
            "abstract": paper["abstract"],
            "authors": paper.get("authors", []),
            "categories": paper.get("categories", []),
            "summary": "该论文总结失败，请直接查看摘要和 arXiv 链接。",
            "chinese_digest": (
                "## 中文导读\n"
                "- 研究问题：LLM 调用失败，未能生成导读。\n"
                "- 核心方法：请参考论文摘要。\n"
                "- 主要结果：请参考论文摘要。\n"
                "- 与量子精密测量/量子计算的关系：请根据摘要判断。\n"
                "- 是否值得精读：请根据标题、摘要和 relevance score 判断。\n"
                "- 对我研究的可能启发：请查看 arXiv 原文。"
            ),
            "english_digest": (
                "## English Guide\n"
                "- Problem: LLM analysis failed; please refer to the abstract.\n"
                "- Method: Please refer to the abstract.\n"
                "- Main result: Please refer to the abstract.\n"
                "- Relevance: Please judge from the title and abstract.\n"
                "- Reading priority: Please judge from the title, abstract, and relevance score.\n"
                "- Possible inspiration: Please open the arXiv link for details."
            ),
            "relevance_score": relevance_score,
            "pdf_url": paper.get("pdf_url", ""),
            "arxiv_url": arxiv_url,
            "local_pdf_path": "",
        }

    def make_pdf_filename(self, paper):
        arxiv_id = re.sub(r"[^A-Za-z0-9._-]+", "_", paper["arXiv_id"]).strip("_")
        short_title = re.sub(r"[^A-Za-z0-9._-]+", "_", paper["title"]).strip("_")
        short_title = re.sub(r"_+", "_", short_title)[:80].strip("_")
        if not short_title:
            short_title = "paper"
        return f"{arxiv_id}__{short_title}.pdf"

    def download_pdf(self, paper, pdf_dir):
        pdf_url = paper.get("pdf_url")
        if not pdf_url:
            logger.warning(f"Skip PDF download for {paper['arXiv_id']}: missing PDF URL.")
            return ""

        os.makedirs(pdf_dir, exist_ok=True)
        pdf_path = os.path.join(pdf_dir, self.make_pdf_filename(paper))
        if os.path.exists(pdf_path):
            return pdf_path

        try:
            response = requests.get(pdf_url, timeout=60)
            response.raise_for_status()
            with open(pdf_path, "wb") as f:
                f.write(response.content)
            return pdf_path
        except Exception as error:
            logger.warning(
                f"Failed to download PDF for {paper['arXiv_id']} from {pdf_url}: {error}"
            )
            return ""

    def download_top_pdfs(self, recommendations):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        pdf_dir = os.path.join(base_dir, "data", "pdfs", self.run_date)
        top_papers = [
            paper for paper in recommendations if paper["relevance_score"] >= 8
        ][:5]

        for paper in top_papers:
            paper["local_pdf_path"] = self.download_pdf(paper, pdf_dir)

        for paper in recommendations:
            paper.setdefault("local_pdf_path", "")

    def process_paper(self, paper, max_retries=5):
        retry_count = 0
        cache_path = os.path.join(self.cache_dir, f"{paper['arXiv_id']}.json")

        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as cache_file:
                    cached_result = json.load(cache_file)
                cached_result.setdefault("authors", paper.get("authors", []))
                cached_result.setdefault("categories", paper.get("categories", []))
                cached_result.setdefault(
                    "arxiv_url",
                    paper.get("abstract_url", f"https://arxiv.org/abs/{paper['arXiv_id']}"),
                )
                print(f"缓存文件 {cache_path} 读取成功。")
                return cached_result
            except (json.JSONDecodeError, OSError) as e:
                print(f"缓存文件 {cache_path} 读取失败: {e}，将重新获取。")

        while retry_count < max_retries:
            try:
                title = paper["title"]
                abstract = paper["abstract"]
                response = self.get_response(title, abstract)
                response = self.clean_model_response(response)
                response = json.loads(response)
                relevance_score = float(
                    response.get("relevance", response.get("relevance_score"))
                )
                summary = response["summary"]
                result = {
                    "title": title,
                    "arXiv_id": paper["arXiv_id"],
                    "abstract": abstract,
                    "authors": paper.get("authors", []),
                    "categories": paper.get("categories", []),
                    "summary": summary,
                    "chinese_digest": response.get("chinese_digest", ""),
                    "english_digest": response.get("english_digest", ""),
                    "relevance_score": relevance_score,
                    "pdf_url": paper["pdf_url"],
                    "arxiv_url": paper.get("abstract_url", f"https://arxiv.org/abs/{paper['arXiv_id']}"),
                    "local_pdf_path": "",
                }
                try:
                    with self.lock:
                        with open(cache_path, "w", encoding="utf-8") as cache_file:
                            json.dump(result, cache_file, ensure_ascii=False, indent=2)
                except OSError as write_error:
                    print(f"写入缓存 {cache_path} 时失败: {write_error}")
                return result
            except Exception as e:
                retry_count += 1
                print(f"处理论文 {paper['arXiv_id']} 时发生错误: {e}")
                print(f"正在进行第 {retry_count} 次重试...")
                if retry_count == max_retries:
                    print(f"已达到最大重试次数 {max_retries}，放弃处理论文{paper['arXiv_id']}")
                    result = self.build_fallback_result(paper)
                    try:
                        with self.lock:
                            with open(cache_path, "w", encoding="utf-8") as cache_file:
                                json.dump(result, cache_file, ensure_ascii=False, indent=2)
                    except OSError as write_error:
                        print(f"写入缓存 {cache_path} 时失败: {write_error}")
                    return result
                time.sleep(1)  # 重试前等待1秒

    def get_recommendation(self):
        recommendations = {}
        for category, papers in self.papers.items():
            for paper in papers:
                recommendations[paper["arXiv_id"]] = paper

        print(
            f"Got {len(recommendations)} non-overlapping papers from yesterday's arXiv."
        )
        papers_to_process = list(recommendations.values())
        if not self.include_seen:
            before_filter = len(papers_to_process)
            papers_to_process = self.history.filter_unseen(papers_to_process)
            skipped_count = before_filter - len(papers_to_process)
            if skipped_count:
                print(f"Skipped {skipped_count} papers already in SQLite history.")

        recommendations_ = []
        print("Performing LLM inference...")

        with ThreadPoolExecutor(self.num_workers) as executor:
            futures = []
            for paper in papers_to_process:
                futures.append(executor.submit(self.process_paper, paper))
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Processing papers",
                unit="paper",
            ):
                result = future.result()
                if result:
                    self.history.mark_processed(result)
                    recommendations_.append(result)

        recommendations_ = sorted(
            recommendations_, key=lambda x: x["relevance_score"], reverse=True
        )[: self.max_paper_num]
        self.download_top_pdfs(recommendations_)
        for paper in recommendations_:
            self.history.mark_processed(paper)

        # Save recommendation to markdown file
        current_time = self.run_datetime
        if self.report_dir:
            os.makedirs(self.report_dir, exist_ok=True)
            save_path = os.path.join(
                self.report_dir, f"{current_time.strftime('%Y-%m-%d')}.md"
            )
        else:
            save_path = os.path.join(
                self.save_dir, self.run_date, f"{current_time.strftime('%Y-%m-%d')}.md"
            )

        def _format_list(value):
            if isinstance(value, list):
                return ", ".join(value) if value else "N/A"
            return value or "N/A"

        def _write_paper(f, paper):
            arxiv_url = paper.get("arxiv_url") or f"https://arxiv.org/abs/{paper['arXiv_id']}"
            f.write(f"### {paper['title']}\n\n")
            f.write(f"- arXiv link: {arxiv_url}\n")
            f.write(f"- PDF link: {paper.get('pdf_url', 'N/A')}\n")
            f.write(f"- Local PDF path: {paper.get('local_pdf_path') or 'N/A'}\n")
            f.write(f"- Authors: {_format_list(paper.get('authors'))}\n")
            f.write(f"- Categories: {_format_list(paper.get('categories'))}\n")
            f.write(f"- Score: {paper['relevance_score']}\n\n")
            f.write(f"#### Abstract\n{paper.get('abstract', '')}\n\n")
            f.write(f"{paper.get('chinese_digest', '')}\n\n")
            f.write(f"{paper.get('english_digest', '')}\n\n")

        groups = [
            ("Top Picks", lambda paper: paper["relevance_score"] >= 8),
            (
                "Worth Skimming",
                lambda paper: 6 <= paper["relevance_score"] < 8,
            ),
            ("Others", lambda paper: paper["relevance_score"] < 6),
        ]

        with open(save_path, "w") as f:
            f.write("# Daily arXiv Papers\n")
            f.write(f"## Date: {current_time.strftime('%Y-%m-%d')}\n")
            f.write(f"## Description: {self.description}\n")
            for group_name, predicate in groups:
                group_papers = [paper for paper in recommendations_ if predicate(paper)]
                f.write(f"\n## {group_name}\n\n")
                if not group_papers:
                    f.write("No papers in this section.\n\n")
                    continue
                for paper in group_papers:
                    _write_paper(f, paper)

        return recommendations_

    def summarize(self, recommendations):
        overview = ""
        for i in range(len(recommendations)):
            overview += f"{i + 1}. {recommendations[i]['title']} - {recommendations[i]['summary']} \n"
        prompt_context = """
            你是一个有帮助的学术研究助手，可以帮助我构建每日论文推荐系统。
            以下是我最近研究领域的描述：
            {}
        """.format(self.description)
        papers_context = """
            以下是我从昨天的 arXiv 爬取的论文，我为你提供了标题和摘要：
            {}
        """.format(overview)
        json_instruction = """
            请务必严格按照以下 JSON 结构返回内容，不要添加额外文本或代码块：
            {{
              "trend_summary": "<总体趋势，用中文,使用 html 的语法，不要使用 markdown 的语法>",
              "recommendations": [
                {{
                  "title": "<论文标题>",
                  "relevance_label": "<高度相关/相关/一般相关>",
                  "recommend_reason": "<为什么值得我读>",
                  "key_contribution": "<一句话概括论文关键贡献>"
                }}
              ],
              "additional_observation": "<补充观察，若无请写‘暂无’>"
            }}

            任务要求：
            1. 给出今天论文体现的整体研究趋势，解释其与我研究兴趣的联系。
            2. 精选最值得我精读的论文（建议返回 3-5 篇，可按实际情况增减），说明推荐理由并突出关键贡献。
            3. 如有需要持续关注或潜在风险的方向，请在补充观察中说明；若没有请写“暂无”。
        """
        html_instruction = """
            请直接输出一段 HTML 片段，严格遵循以下结构，不要包含 JSON、Markdown 或多余说明：
            <div class="summary-wrapper">
              <div class="summary-section">
                <h2>今日研究趋势</h2>
                <p>...</p>
              </div>
              <div class="summary-section">
                <h2>重点推荐</h2>
                <ol class="summary-list">
                  <li class="summary-item">
                    <div class="summary-item__header"><span class="summary-item__title">论文标题</span><span class="summary-pill">相关性</span></div>
                    <p><strong>推荐理由：</strong>...</p>
                    <p><strong>关键贡献：</strong>...</p>
                  </li>
                </ol>
              </div>
              <div class="summary-section">
                <h2>补充观察</h2>
                <p>暂无或其他补充。</p>
              </div>
            </div>

            HTML 要用中文撰写内容，重点推荐部分建议返回 3-5 篇论文，可按实际情况增减，缺少推荐时请写“暂无推荐。”。
        """
        prompt = prompt_context + papers_context + json_instruction
        html_prompt = prompt_context + papers_context + html_instruction

        def _clean_model_response(raw_text: str) -> str:
            cleaned = raw_text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()
                if "\n" in cleaned:
                    first_line, rest = cleaned.split("\n", 1)
                    if first_line.strip().lower() in ("json", "html"):
                        cleaned = rest
                    else:
                        cleaned = first_line + "\n" + rest
            return cleaned.strip()

        max_retries = 1
        for attempt in range(1, max_retries + 1):
            try:
                raw_response = self.model.inference(
                    prompt, temperature=self.temperature
                )
                cleaned = _clean_model_response(raw_response)
                data = json.loads(cleaned)
                trend_summary = data.get("trend_summary", "暂无趋势信息")
                recommendations_data = data.get("recommendations", [])
                additional_observation = data.get("additional_observation", "暂无")

                if not isinstance(recommendations_data, list):
                    raise ValueError("recommendations 字段不是列表")

                cleaned_recommendations = []
                for item in recommendations_data:
                    title = item.get("title")
                    if not title:
                        raise ValueError("recommendations 中存在缺少标题的条目")
                    cleaned_recommendations.append(
                        {
                            "title": title,
                            "relevance_label": item.get(
                                "relevance_label", "相关性未知"
                            ),
                            "recommend_reason": item.get(
                                "recommend_reason", "未提供推荐理由"
                            ),
                            "key_contribution": item.get(
                                "key_contribution", "未提供关键贡献"
                            ),
                        }
                    )

                structured_summary = {
                    "trend_summary": trend_summary,
                    "recommendations": cleaned_recommendations,
                    "additional_observation": additional_observation,
                }

                return render_summary_sections(structured_summary)
            except Exception as error:
                print(f"总结生成第 {attempt} 次失败: {error}")
                if attempt == max_retries:
                    try:
                        for html_attempt in range(1, max_retries + 1):  
                            print(f"HTML 回退生成第 {html_attempt} 次...")
                            raw_html_response = self.model.inference(
                                html_prompt, temperature=self.temperature
                            )
                            cleaned_html = _clean_model_response(raw_html_response)
                            return cleaned_html
                    except Exception as html_error:
                        print(f"HTML 回退生成失败: {html_error}")
                        fallback_data = {
                            "trend_summary": "总结生成失败，请稍后重试。",
                            "recommendations": [],
                            "additional_observation": "暂无。",
                        }
                        return render_summary_sections(fallback_data)

    def render_email(self, recommendations):
        save_file_path = os.path.join(self.save_dir, self.run_date, "arxiv_daily_email.html")
        if os.path.exists(save_file_path):
            with open(save_file_path, "r", encoding="utf-8") as f:
                print(f"邮件已渲染，从缓存文件 {save_file_path} 读取邮件。")
                return f.read()
        parts = []
        if len(recommendations) == 0:
            return framework.replace("__CONTENT__", get_empty_html())
        for i, p in enumerate(tqdm(recommendations, desc="Rendering Emails")):
            rate = get_stars(p["relevance_score"])
            parts.append(
                get_block_html(
                    str(i + 1) + ". " + p["title"],
                    rate,
                    p["arXiv_id"],
                    p["summary"],
                    p["pdf_url"],
                    p.get("chinese_digest", ""),
                    p.get("english_digest", ""),
                    p.get("arxiv_url", ""),
                    p.get("abstract", ""),
                )
            )
        summary = self.summarize(recommendations)
        # Add the summary to the start of the email
        content = summary
        content += "<br>" + "</br><br>".join(parts) + "</br>"
        email_html = framework.replace("__CONTENT__", content)
        # 保存渲染后的邮件到 save_dir
        if self.save_dir:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            save_path = os.path.join(base_dir, self.save_dir, self.run_date, "arxiv_daily_email.html")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(email_html)
        return email_html

    def send_email(
        self,
        sender: str,
        receiver: str,
        password: str,
        smtp_server: str,
        smtp_port: int,
        title: str,
    ):
        recommendations = self.get_recommendation()
        html = self.render_email(recommendations)

        def _format_addr(s):
            name, addr = parseaddr(s)
            return formataddr((Header(name, "utf-8").encode(), addr))

        msg = MIMEText(html, "html", "utf-8")
        msg["From"] = _format_addr(f"{title} <%s>" % sender)

        # 处理多个接收者
        receivers = [addr.strip() for addr in receiver.split(",")]
        print(receivers)
        msg["To"] = ",".join([_format_addr(f"You <%s>" % addr) for addr in receivers])

        today = self.run_datetime.strftime("%Y/%m/%d")
        msg["Subject"] = Header(f"{title} {today}", "utf-8").encode()

        try:
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
        except Exception as e:
            logger.warning(f"Failed to use TLS. {e}")
            logger.warning(f"Try to use SSL.")
            server = smtplib.SMTP_SSL(smtp_server, smtp_port)

        server.login(sender, password)
        server.sendmail(sender, receivers, msg.as_string())
        server.quit()


if __name__ == "__main__":
    categories = ["cs.CV"]
    max_entries = 100
    max_paper_num = 50
    provider = "ollama"
    model = "deepseek-r1:7b"
    description = """
        I am working on the research area of computer vision and natural language processing. 
        Specifically, I am interested in the following fieds:
        1. Object detection
        2. AIGC (AI Generated Content)
        3. Multimodal Large Language Models

        I'm not interested in the following fields:
        1. 3D Vision
        2. Robotics
        3. Low-level Vision
    """

    arxiv_daily = ArxivDaily(
        categories, max_entries, max_paper_num, provider, model, None, None, description
    )
    recommendations = arxiv_daily.get_recommendation()
    print(recommendations)
