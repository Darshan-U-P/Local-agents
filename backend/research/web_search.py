from __future__ import annotations

import html
import re
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from dataclasses import dataclass


# ============================================================
# Search result
# ============================================================

@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


# ============================================================
# HTML text parser
# ============================================================

class TextParser(HTMLParser):

    SKIP_TAGS = {
        "script",
        "style",
        "noscript",
        "svg",
        "nav",
        "footer",
        "header",
        "form",
    }

    def __init__(self):
        super().__init__()

        self.text_parts = []
        self.skip_depth = 0

    def handle_starttag(self, tag, attrs):

        tag = tag.lower()

        if tag in self.SKIP_TAGS:
            self.skip_depth += 1

    def handle_endtag(self, tag):

        tag = tag.lower()

        if tag in self.SKIP_TAGS and self.skip_depth > 0:
            self.skip_depth -= 1

    def handle_data(self, data):

        if self.skip_depth > 0:
            return

        text = data.strip()

        if text:
            self.text_parts.append(text)

    def get_text(self):

        text = " ".join(self.text_parts)

        text = html.unescape(text)

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()


# ============================================================
# DuckDuckGo search parser
# ============================================================

class SearchParser(HTMLParser):

    def __init__(self):

        super().__init__()

        self.results = []

        self.current_result = None

        self.in_result_title = False
        self.in_result_snippet = False

    def handle_starttag(self, tag, attrs):

        attrs = dict(attrs)

        classes = attrs.get("class", "")

        # DuckDuckGo result title
        if (
            tag == "a"
            and "result__a" in classes
        ):

            href = attrs.get("href", "")

            self.current_result = {
                "title": "",
                "url": href,
                "snippet": "",
            }

            self.in_result_title = True

        # DuckDuckGo result snippet
        elif (
            self.current_result
            and "result__snippet" in classes
        ):

            self.in_result_snippet = True

    def handle_endtag(self, tag):

        if tag == "a" and self.in_result_title:
            self.in_result_title = False

        if self.in_result_snippet:
            self.in_result_snippet = False

            if self.current_result:

                self.results.append(
                    self.current_result
                )

                self.current_result = None

    def handle_data(self, data):

        if self.current_result is None:
            return

        data = data.strip()

        if not data:
            return

        if self.in_result_title:
            self.current_result["title"] += data

        elif self.in_result_snippet:
            self.current_result["snippet"] += data


# ============================================================
# Web Search
# ============================================================

class WebSearch:

    USER_AGENT = (
        "Mozilla/5.0 "
        "(Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/154.0 Safari/537.36"
    )

    def search(
        self,
        query: str,
        max_results: int = 8,
    ) -> list[SearchResult]:

        if not query.strip():
            raise ValueError(
                "Search query cannot be empty."
            )

        encoded_query = urllib.parse.quote_plus(
            query
        )

        url = (
            "https://html.duckduckgo.com/html/"
            f"?q={encoded_query}"
        )

        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": self.USER_AGENT,
            },
        )

        print(
            f"Searching web: {query}"
        )

        with urllib.request.urlopen(
            request,
            timeout=20,
        ) as response:

            content = response.read().decode(
                "utf-8",
                errors="ignore",
            )

        parser = SearchParser()

        parser.feed(content)

        results = []

        for item in parser.results:

            title = html.unescape(
                item["title"]
            ).strip()

            result_url = item["url"].strip()

            snippet = html.unescape(
                item["snippet"]
            ).strip()

            if not title or not result_url:
                continue

            # Resolve DuckDuckGo redirect URLs.
            result_url = self._resolve_url(
                result_url
            )

            results.append(
                SearchResult(
                    title=title,
                    url=result_url,
                    snippet=snippet,
                )
            )

            if len(results) >= max_results:
                break

        return results

    @staticmethod
    def _resolve_url(url: str) -> str:

        # DuckDuckGo sometimes returns:
        #
        # /l/?uddg=https%3A%2F%2Fexample.com

        if "uddg=" in url:

            parsed = urllib.parse.urlparse(
                url
            )

            params = urllib.parse.parse_qs(
                parsed.query
            )

            if "uddg" in params:
                return params["uddg"][0]

        return url


# ============================================================
# Page fetching
# ============================================================

class WebPageFetcher:

    USER_AGENT = WebSearch.USER_AGENT

    def fetch(
        self,
        url: str,
        timeout: int = 20,
    ) -> str:

        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": self.USER_AGENT,
            },
        )

        with urllib.request.urlopen(
            request,
            timeout=timeout,
        ) as response:

            content_type = response.headers.get(
                "Content-Type",
                "",
            )

            # Only process HTML pages.
            if "text/html" not in content_type.lower():

                return ""

            raw = response.read()

        parser = TextParser()

        parser.feed(
            raw.decode(
                "utf-8",
                errors="ignore",
            )
        )

        text = parser.get_text()

        # Prevent enormous pages from entering Qwen.
        return text[:30000]