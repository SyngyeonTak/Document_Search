import urllib3
from opensearchpy import OpenSearch
import os

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def get_client():

    host = os.getenv("OPENSEARCH_HOST", "")
    port = os.getenv("OPENSEARCH_PORT", "")
    user = os.getenv("OPENSEARCH_USER", "")
    password = os.getenv("OPENSEARCH_PASSWORD", "")
    auth = (user, password)

    client = OpenSearch(
        hosts=[{'host':host, 'port':port}]
        , http_compress = True
        , http_auth = auth
        , use_ssl = True
        , verify_certs=False
        , ssl_assert_hostname=False
        , ssl_show_warn=False
    )

    return client

