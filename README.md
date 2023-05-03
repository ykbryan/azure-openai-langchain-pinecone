### Install

```
!python --version
!pip install langchain --upgrade
# !conda install langchain -c conda-forge --y
# Version: 0.0.149
```

```
!pip install openai --upgrade
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
CHAT_MODEL = ''
```

### Load your data

```
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
```

```
# loader = UnstructuredPDFLoader("../data/some.pdf")
loader = OnlinePDFLoader("https://wolfpaulus.com/wp-content/uploads/2017/05/field-guide-to-data-science.pdf")
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

```
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(data)
```

```
print (f'Now you have {len(texts)} documents')
```

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
# initialize pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_API_ENV  # next to api key in console
)
index_name = "langchaindemo" # put in the name of your pinecone index here
```

```
# !pip install tiktoken
```

```
docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)
```

```
query = "What are examples of good data science teams?"
docs = docsearch.similarity_search(query, include_metadata=True)
```

```
# Here's an example of the first document that was returned
docs[0].page_content[:250]
```

### Query those docs to get your answer back

```
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
```

```
llm = OpenAI(
    temperature=0,
    openai_api_key=OPENAI_API_KEY,
    openai_api_base=OPENAI_API_BASE,
    model_name = "text-davinci-003",
    deployment_id = "davinci",
    max_tokens=100
)
chain = load_qa_chain(llm, chain_type="stuff")
```

```
query = "What is the collect stage of data maturity?"
docs = docsearch.similarity_search(query, include_metadata=True)
```

```
# Run the LLM
llm("Tell me a joke")
```

```
chain.run(input_documents=docs, question=query)
```
