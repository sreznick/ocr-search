from tqdm.notebook import tqdm

import djvu.decode

def print_text(sexpr, level=0):
    if level > 0:
        print(' ' * (2 * level - 1), end=' ')
    if isinstance(sexpr, djvu.sexpr.ListExpression):
        if len(sexpr) == 0:
            return
        print(str(sexpr[0].value), [sexpr[i].value for i in range(1, 5)])
        for child in sexpr[5:]:
            print_text(child, level + 1)
    else:
        print(sexpr)

class Context(djvu.decode.Context):

    def process(self, djvu_path, pages=[]):
        document = self.new_document(djvu.decode.FileURI(djvu_path))
        for i, page in tqdm(enumerate(document.pages)):
            page.get_info(wait=True)
            if i not in pages and pages != []:
                continue
            print_text(page.text.sexpr)

def dump_text(djvu_path, pages=[]):
    context = Context()
    context.process(djvu_path, pages)