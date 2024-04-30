# CHAT WITH DOCS

#### Baseado nas soluções (https://github.com/imartinez/privateGPT), (https://github.com/jmorganca/ollama) e (https://github.com/PromptEngineer48/Ollama)

#### Requisitos
```
pip install -r requirements.txt
```

#### Models a serem utilizados (Meus testes foram feitos com Mistral, Llama2 e/ou Llama3)
#### Caso não tenho o Ollama instalado -> https://ollama.ai
#### Exemplo:
```
ollama pull mistral
```

#### Crie o diretório para incluir os arquivos a serem lidos e avaliados
```
mkdir source_documents
```

#### Realize a avaliação dos arquivos usando o código abaixo
```
python ingest.py
```

O output deve ser +/- assim:
```shell
Creating new vectorstore
Loading documents from source_documents
Loading new documents: 100%|██████████████████████| 1/1 [00:01<00:00,  1.99s/it]
Loaded 235 new documents from source_documents
Split into 1268 chunks of text (max. 500 tokens each)
Creating embeddings. May take some minutes...
Ingestion complete! You can now run privateGPT.py to query your documents
```

#### Rode o programa para conversa sobre o arquivo (testado apenas em inglês)
```
python privateGPT.py
```

##### Insira a questão
Enter a query: Who is Thiago Seiki Kato?


### Tente com Llama2 ou outro modelo:
```
ollama pull llama2
MODEL=llama2 python privateGPT.py
```

## Adicione mais arquivos

Insira os arquivos no diretório criado `source_documents`

As extensões suportadas são:

- `.csv`: CSV,
- `.docx`: Word Document,
- `.doc`: Word Document,
- `.enex`: EverNote,
- `.eml`: Email,
- `.epub`: EPub,
- `.html`: HTML File,
- `.md`: Markdown,
- `.msg`: Outlook Message,
- `.odt`: Open Document Text,
- `.pdf`: Portable Document Format (PDF),
- `.pptx` : PowerPoint Document,
- `.ppt` : PowerPoint Document,
- `.txt`: Text file (UTF-8),
- `.json`: Json file (UTF-8),

- testado com PDF, JSON e DOCX
