from lxml import etree

def fix_unbound_prefix(path):
    parser = etree.XMLParser(encoding="utf-8", recover=True)
    tree = etree.parse(path, parser)

    return tree