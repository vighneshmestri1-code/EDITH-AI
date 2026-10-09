import ipaddress
import socket
import webbrowser
from urllib.parse import quote_plus, urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from tools.registry import register_tool
from utils.logger import logger


try:
    from ddgs import DDGS

    SEARCH_BACKEND = "ddgs"

except ImportError:

    try:
        from duckduckgo_search import DDGS

        SEARCH_BACKEND = "duckduckgo_search"

    except ImportError:

        DDGS = None
        SEARCH_BACKEND = None


MAX_SEARCH_RESULTS = 10
MAX_WEBPAGE_BYTES = 2 * 1024 * 1024
MAX_REDIRECTS = 5


def _validate_query(
    query: str,
) -> str | None:
    """Validate and normalize a search query."""

    if not query:
        return None

    cleaned_query = str(
        query
    ).strip()

    if not cleaned_query:
        return None

    return cleaned_query


def _normalize_max_results(
    max_results: int,
) -> int:
    """Keep search result count within a safe range."""

    try:
        max_results = int(
            max_results
        )

    except (TypeError, ValueError):
        max_results = 5

    return max(
        1,
        min(
            max_results,
            MAX_SEARCH_RESULTS,
        ),
    )


def _normalize_url(
    url: str,
) -> str:
    """
    Normalize a URL and require HTTP or HTTPS.
    """

    if not url or not str(url).strip():
        raise ValueError(
            "URL cannot be empty."
        )

    url = str(url).strip()

    if not url.startswith(
        (
            "http://",
            "https://",
        )
    ):
        url = f"https://{url}"

    parsed = urlparse(
        url
    )

    if parsed.scheme not in {
        "http",
        "https",
    }:
        raise ValueError(
            "Only HTTP and HTTPS URLs are allowed."
        )

    if not parsed.hostname:
        raise ValueError(
            "URL must contain a valid hostname."
        )

    return url


def _is_private_ip(
    ip: str,
) -> bool:
    """Return True when an IP belongs to a non-public range."""

    try:
        address = ipaddress.ip_address(
            ip
        )

    except ValueError:
        return False

    return (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    )


def _validate_public_hostname(
    hostname: str,
) -> None:
    """
    Reject localhost and hostnames that resolve to private
    or otherwise non-public IP addresses.
    """

    normalized_host = hostname.strip().lower()

    blocked_hostnames = {
        "localhost",
        "localhost.localdomain",
    }

    if normalized_host in blocked_hostnames:
        raise PermissionError(
            "Access to local hosts is not allowed."
        )

    try:
        direct_ip = ipaddress.ip_address(
            normalized_host
        )

        if _is_private_ip(
            str(direct_ip)
        ):
            raise PermissionError(
                "Access to private or local "
                "IP addresses is not allowed."
            )

        return

    except ValueError:
        pass

    try:
        addresses = socket.getaddrinfo(
            normalized_host,
            None,
            type=socket.SOCK_STREAM,
        )

    except socket.gaierror as e:
        raise ValueError(
            f"Unable to resolve hostname '{hostname}'."
        ) from e

    resolved_ips = {
        result[4][0]
        for result in addresses
        if result and result[4]
    }

    for ip in resolved_ips:
        if _is_private_ip(
            ip
        ):
            raise PermissionError(
                "Access to private or local "
                "network addresses is not allowed."
            )


def _validate_fetch_url(
    url: str,
) -> str:
    """Validate a webpage URL before making a request."""

    normalized_url = _normalize_url(
        url
    )

    parsed = urlparse(
        normalized_url
    )

    _validate_public_hostname(
        parsed.hostname or ""
    )

    return normalized_url


@register_tool(name="web_search")
def web_search(
    query: str,
    max_results: int = 5,
) -> str:
    """
    Search the live web and return readable search results.

    This tool retrieves actual search results for EDITH.
    """

    query = _validate_query(
        query
    )

    if not query:
        return "Please provide a search query."

    if DDGS is None:

        logger.error(
            "No DuckDuckGo search package is installed."
        )

        return (
            "Web search is unavailable. "
            "Install the 'ddgs' package with "
            "'pip install -U ddgs'."
        )

    max_results = _normalize_max_results(
        max_results
    )

    logger.info(
        f"Web search started: '{query}' "
        f"(max_results={max_results}, "
        f"backend={SEARCH_BACKEND})"
    )

    try:
        search_results = []

        with DDGS() as ddgs:

            results = ddgs.text(
                query,
                max_results=max_results,
            )

            for index, result in enumerate(
                results or [],
                start=1,
            ):

                if not isinstance(
                    result,
                    dict,
                ):
                    continue

                title = str(
                    result.get(
                        "title",
                        "",
                    )
                ).strip()

                snippet = str(
                    result.get(
                        "body",
                        result.get(
                            "description",
                            "",
                        ),
                    )
                ).strip()

                url = str(
                    result.get(
                        "href",
                        result.get(
                            "url",
                            "",
                        ),
                    )
                ).strip()

                if not title and not snippet and not url:
                    continue

                search_results.append(
                    f"{index}. {title or 'Untitled'}\n"
                    f"   Snippet: "
                    f"{snippet or 'No snippet available.'}\n"
                    f"   URL: "
                    f"{url or 'No URL available.'}"
                )

        if not search_results:

            logger.warning(
                f"No web results found for '{query}'."
            )

            return (
                f"No results found for query: "
                f"'{query}'"
            )

        logger.info(
            f"Web search completed: '{query}' "
            f"({len(search_results)} results)"
        )

        return "\n\n".join(
            search_results
        )

    except Exception as e:

        logger.error(
            f"Web search failed for "
            f"'{query}': {e}",
            exc_info=True,
        )

        return (
            f"Error performing web search: "
            f"{str(e)}"
        )


@register_tool(name="browser_search")
def browser_search(
    query: str,
    engine: str = "google",
) -> str:
    """
    Open a browser search page.

    This tool navigates the browser; it does not retrieve
    and summarize search results for EDITH.
    """

    query = _validate_query(
        query
    )

    if not query:
        return "Please provide a search query."

    engine = str(
        engine
    ).strip().lower()

    encoded_query = quote_plus(
        query
    )

    search_urls = {
        "google": (
            "https://www.google.com/search?q="
            f"{encoded_query}"
        ),
        "bing": (
            "https://www.bing.com/search?q="
            f"{encoded_query}"
        ),
        "duckduckgo": (
            "https://duckduckgo.com/?q="
            f"{encoded_query}"
        ),
        "youtube": (
            "https://www.youtube.com/results"
            f"?search_query={encoded_query}"
        ),
    }

    if engine not in search_urls:

        supported = ", ".join(
            search_urls.keys()
        )

        return (
            f"Unsupported search engine "
            f"'{engine}'. "
            f"Supported engines: {supported}."
        )

    url = search_urls[
        engine
    ]

    try:

        opened = webbrowser.open(
            url
        )

        if not opened:

            logger.warning(
                f"Browser could not open "
                f"{engine} search for '{query}'."
            )

            return (
                f"I couldn't open the browser "
                f"for the {engine} search."
            )

        logger.info(
            f"Browser search opened: "
            f"{engine} → '{query}'"
        )

        return (
            f"Searching {engine.title()} for "
            f"'{query}' in your browser."
        )

    except Exception as e:

        logger.error(
            f"Browser search failed for "
            f"'{query}': {e}",
            exc_info=True,
        )

        return (
            f"Error opening browser search: "
            f"{str(e)}"
        )


@register_tool(name="browser_open_url")
def browser_open_url(
    url: str,
) -> str:
    """Open a specific URL in the user's default browser."""

    if not url or not str(url).strip():
        return "Please provide a URL."

    try:

        url = _normalize_url(
            url
        )

        opened = webbrowser.open(
            url
        )

        if not opened:

            return (
                f"I couldn't open the URL: "
                f"{url}"
            )

        logger.info(
            f"Browser URL opened: {url}"
        )

        return (
            f"Opening {url} in your browser."
        )

    except Exception as e:

        logger.error(
            f"Error opening URL "
            f"'{url}': {e}",
            exc_info=True,
        )

        return (
            f"Error opening URL: {str(e)}"
        )


@register_tool(name="fetch_webpage")
def fetch_webpage(
    url: str,
) -> str:
    """Fetch and extract readable text from a web page."""

    try:
        current_url = _validate_fetch_url(
            url
        )

        headers = {
            "User-Agent": (
                "Mozilla/5.0 "
                "(Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/154.0.0.0 Safari/537.36"
            )
        }

        with httpx.Client(
            timeout=15.0,
            follow_redirects=False,
            headers=headers,
            trust_env=False,
        ) as client:

            response = None

            for redirect_count in range(
                MAX_REDIRECTS + 1
            ):

                current_url = _validate_fetch_url(
                    current_url
                )

                with client.stream(
                    "GET",
                    current_url,
                ) as stream:

                    if (
                        300
                        <= stream.status_code
                        < 400
                    ):
                        location = (
                            stream.headers.get(
                                "location"
                            )
                        )

                        if not location:
                            return (
                                "The webpage returned "
                                "a redirect without a "
                                "destination."
                            )

                        current_url = urljoin(
                            current_url,
                            location,
                        )

                        continue

                    stream.raise_for_status()

                    content_length = (
                        stream.headers.get(
                            "content-length"
                        )
                    )

                    if content_length:

                        try:
                            declared_size = int(
                                content_length
                            )

                        except ValueError:
                            declared_size = 0

                        if declared_size > MAX_WEBPAGE_BYTES:
                            return (
                                "The webpage is too large "
                                "to process."
                            )

                    body = bytearray()

                    for chunk in stream.iter_bytes(
                        chunk_size=64 * 1024
                    ):

                        body.extend(
                            chunk
                        )

                        if len(body) > MAX_WEBPAGE_BYTES:
                            return (
                                "The webpage is too large "
                                "to process."
                            )

                    response = stream

                    encoding = (
                        response.encoding
                        or "utf-8"
                    )

                    raw_content = bytes(
                        body
                    )

                    try:
                        page_text = (
                            raw_content.decode(
                                encoding,
                                errors="replace",
                            )
                        )

                    except LookupError:
                        page_text = (
                            raw_content.decode(
                                "utf-8",
                                errors="replace",
                            )
                        )

                    break

            else:
                return (
                    "The webpage exceeded the "
                    "maximum redirect limit."
                )

        if response is None:
            return (
                "Unable to retrieve webpage content."
            )

        content_type = (
            response.headers.get(
                "content-type",
                "",
            ).lower()
        )

        allowed_content_types = (
            "text/html",
            "application/xhtml+xml",
            "text/plain",
        )

        if content_type and not any(
            content_type.startswith(
                allowed
            )
            for allowed in allowed_content_types
        ):
            return (
                "The URL did not return a supported "
                "text or HTML webpage."
            )

        soup = BeautifulSoup(
            page_text,
            "html.parser",
        )

        for element in soup(
            [
                "script",
                "style",
                "nav",
                "footer",
                "header",
                "noscript",
            ]
        ):
            element.decompose()

        text = soup.get_text(
            separator="\n",
            strip=True,
        )

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        clean_text = "\n".join(
            lines
        )

        if not clean_text:
            return (
                "The webpage did not contain "
                "readable text."
            )

        if len(clean_text) > 5000:
            clean_text = (
                clean_text[:5000]
                + "\n...[Content Truncated]"
            )

        logger.info(
            f"Successfully fetched webpage: "
            f"{current_url}"
        )

        return clean_text

    except Exception as e:

        logger.error(
            f"Error fetching webpage "
            f"'{url}': {e}",
            exc_info=True,
        )

        return (
            f"Error fetching webpage: "
            f"{str(e)}"
        )