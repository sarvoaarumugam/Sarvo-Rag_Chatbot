from strands import tool
from ddgs import DDGS
from ddgs.exceptions import RatelimitException, DDGSException


@tool
async def websearch(query: str, max_results: int = 5) -> str:
    """
    Search the web using DuckDuckGo for current information.
    
    Args:
        query: The search query string
        max_results: Maximum number of results to return (default: 5)
    
    Returns:
        A formatted string with search results
    """
    try:
        # Create DDGS instance
        ddgs = DDGS()
        
        # Perform search
        results = []
        try:
            # Using text search for general queries
            search_results = ddgs.text(
                keywords=query,
                max_results=max_results,
                safesearch='moderate',
                backend='api'  # Using API backend for better reliability
            )
            
            for result in search_results:
                results.append({
                    'title': result.get('title', 'No title'),
                    'body': result.get('body', 'No description'),
                    'url': result.get('href', 'No URL')
                })
                
        except RatelimitException:
            return "Rate limit reached. Please try again in a few moments."
        except DDGSException as e:
            return f"Search error: {str(e)}"
        except Exception as e:
            return f"Unexpected error during search: {str(e)}"
        
        if not results:
            return f"No results found for query: '{query}'"
        
        # Format results
        formatted_results = f"Search results for '{query}':\n\n"
        for i, result in enumerate(results, 1):
            formatted_results += f"{i}. **{result['title']}**\n"
            formatted_results += f"   {result['body']}\n"
            formatted_results += f"   URL: {result['url']}\n\n"
        
        return formatted_results
        
    except Exception as e:
        return f"Error performing web search: {str(e)}"
