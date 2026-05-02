from tree_sitter import Language, Parser
import tree_sitter_javascript as ts_js


parser = Parser()
parser.set_language(ts_js.language())


def parse_js(code: str):
    tree = parser.parse(bytes(code, "utf8"))
    return tree