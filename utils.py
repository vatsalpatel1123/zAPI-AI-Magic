from datetime import datetime
import re
import uuid

def generate_unique_name(url: str) -> str:
    """Generate a unique name for the folder based on the URL."""
    timestamp = datetime.now().strftime('%Y_%m_%d__%H_%M_%S_%f')
    domain = re.sub(r'\W+', '_', url.split('//')[-1].split('/')[0])
    return f"{domain}_{timestamp}"

def generate_api_link() -> str:
    """Generates a unique API link using UUID."""
    return str(uuid.uuid4())