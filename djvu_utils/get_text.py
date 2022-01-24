from tqdm.notebook import tqdm

import djvu.decode

def print_text(sexpr, text_file, level=0):
    if level > 0:
        text_file.write(' ' * (2 * level - 1) + ' ')
    if isinstance(sexpr, djvu.sexpr.ListExpression):
        if len(sexpr) == 0:
            return
        text_file.write(str(sexpr[0].value))
        text_file.write(str([sexpr[i].value for i in range(1, 5)]) + "\n")
        for child in sexpr[5:]:
            print_text(child, text_file, level + 1)
    else:
        text_file.write(str(sexpr) + "\n")

class Context(djvu.decode.Context):

    def process(self, djvu_path, text_file, pages=[]):
        document = self.new_document(djvu.decode.FileURI(djvu_path))
        for i, page in tqdm(enumerate(document.pages)):
            page.get_info(wait=True)
            if i not in pages and pages != []:
                continue
            print_text(page.text.sexpr, text_file)

def dump_text(djvu_path, text_file_path, pages=[]):
    text_file = open(text_file_path, 'w')
    context = Context()
    context.process(djvu_path, text_file, pages)