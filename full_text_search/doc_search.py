from elasticsearch import Elasticsearch


def insert_new_doc_for_es(book_id, page_number, line_number, text, new_doc_id, index='test'):
    new_doc = {
        "book_id": book_id,
        "page_number": page_number,
        "line_number": line_number,
        "text": text,
    }
    es = Elasticsearch([{'host':'localhost','port':9200}])
    res = es.index(index=index, id=new_doc_id, document=new_doc)
    return res['result']


def create_docs_from_book(path_to_book, book_id, last_doc_id=0):
    page_number = 0
    line_number = 0
    text = ""
    new_doc_id = last_doc_id + 1
    with open(path_to_book) as input_file:
        for line in input_file:
            if line.strip().startswith('page['):
                page_number += 1
                line_number = 0
            elif line.strip().startswith('line['):
                if line_number:
                    insert_new_doc_for_es(
                        book_id,
                        page_number,
                        line_number, text,
                        new_doc_id,
                    )
                    new_doc_id += 1
                line_number += 1
                text = ""
            elif not line.strip().startswith('word['):
                text += (line.strip()[1:-1] + ' ')
           
                
def full_text_search(query, index='test'):
    es = Elasticsearch([{'host':'localhost','port':9200}])
    res = es.search(
        index=index,
        query={
            'match': {
                'text': query,
            }
        }
    )
    return res['hits']['hits']

