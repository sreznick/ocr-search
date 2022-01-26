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


def create_docs_from_book(path_to_book, book_id, last_doc_id=0, index='test'):
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
                        index=index,
                    )
                    new_doc_id += 1
                line_number += 1
                text = ""
            elif not line.strip().startswith('word['):
                text += (line.strip()[1:-1] + ' ')
    return new_doc_id - 1
           
                
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


def get_beautiful_search_results(query, file_with_book_ids, index='test'):
    titles_from_ids_dict = {}
    with open(file_with_book_ids, 'r') as book_ids_file:
        for line in book_ids_file:
            curr_id = line.split()[0]
            book_title = line.replace(curr_id, '', 1).strip()
            titles_from_ids_dict[curr_id] = book_title
    search_results = full_text_search(query, index=index)
    beautiful_results = []
    for res in search_results:
        book_title = titles_from_ids_dict[str(res['_source']['book_id'])]
        text = res['_source']['text'].strip()
        page = str(res['_source']['page_number'])
        line = str(res['_source']['line_number'])
        curr_res = '"{}" из книги "{}", страница {}, строка {}'.format(text, book_title, page, line)
        beautiful_results.append(curr_res)
    return beautiful_results
        

def get_beautiful_serp(query, file_with_book_ids, index='test'):
    results_list = get_beautiful_search_results(query, file_with_book_ids, index=index)
    ans = 'По вашему запросу найдено:\n'
    for i, curr_res in enumerate(results_list):
        ans += (str(i + 1) + '. ' + curr_res + '\n')
    return ans
