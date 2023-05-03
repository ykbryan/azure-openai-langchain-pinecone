### Install langchain and OpenAI

```
# !python --version
# !pip install langchain --upgrade
# !conda install langchain -c conda-forge --y
# Version: 0.0.149
```

```
# !pip install openai --upgrade
# !conda install openai -c conda-forge --y
```

### Prepare your keys

```
OPENAI_API_KEY = ''
OPENAI_API_BASE = 'https://xxxxxx.openai.azure.com'
OPENAI_API_TYPE = 'azure'
OPENAI_API_VERSION = '2022-12-01' # 2023-03-15-preview
PINECONE_API_KEY = ''
PINECONE_API_ENV = ''
```

### Load your data

```
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
```

```
# loader = UnstructuredPDFLoader("../data/some.pdf")
loader = OnlinePDFLoader("https://github.com/Azure-Samples/azure-search-openai-demo/raw/main/data/Northwind_Standard_Benefits_Details.pdf")
```

```
# !pip install unstructured
# !pip install pdfminer.six
```

```
data = loader.load()
```

```
print (f'You have {len(data)} document(s) in your data')
print (f'There are {len(data[0].page_content)} characters in your document')
```

### Splitting documents into chunks

```
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(data)
```

```
print (f'Now you have {len(texts)} documents')
```

### Initialize Pinecone

```
# !pip install pinecone-client
```

```
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
```

```
embeddings = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY,
    openai_api_base=OPENAI_API_BASE,
    openai_api_type=OPENAI_API_TYPE,
    openai_api_version=OPENAI_API_VERSION,
    chunk_size=1
)
```

```
pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_API_ENV  # next to api key in console
)
index_name = "langchaindemo" # put in the name of your pinecone index here
```

### Storing documents and embeddings in a vectorstore

```
# !pip install tiktoken
```

```
docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)
```

### Do a search on Pinecone

```
query = "what is included in my health benefit"
docs = docsearch.similarity_search(query, include_metadata=True)
```

```
# Here's an example of the first document that was returned
docs[0].page_content[:250]
```

### Use Azure OpenAI

```
from langchain.llms import AzureOpenAI
# from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
```

```
llm = AzureOpenAI(
    deployment_name="davinci",
    model_name="text-davinci-003",
    openai_api_key=OPENAI_API_KEY
)

# llm = OpenAI(
#    temperature=0,
#    openai_api_key=OPENAI_API_KEY,
#    openai_api_base=OPENAI_API_BASE,
#    model_name = "text-davinci-003",
#    deployment_id = "davinci",
#    max_tokens=100
#)
```

```
# Remember you did a test search before?
# docs = docsearch.similarity_search(query, include_metadata=True)
```

```
# Run the LLM
# print(llm(query))
```

### Chain them

```
chain = load_qa_chain(llm, chain_type="stuff")
chain.run(input_documents=docs, question=query)
```
