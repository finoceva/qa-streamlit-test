from llama_index.node_parser import SimpleNodeParser
from llama_index.data_structs import Node
from llama_index.schema import IndexNode
from llama_index import Document
from typing import List


def simple_convert_documents_into_nodes(
    documents: List[Document],
    chunk_size=512,
    chunk_overlap=20,
    re_name_node_ids: bool = True,
) -> List[Node]:
    node_parser = SimpleNodeParser.from_defaults(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    nodes = node_parser.get_nodes_from_documents(documents)
    # set node ids to be a constant
    if re_name_node_ids:
        for idx, node in enumerate(nodes):
            node.id_ = f"node-{idx}"
    return nodes


def small_to_big_conversion(
    base_nodes: List[Node],
    sub_chunk_sizes: List[int],
) -> List[Node]:
    sub_node_parsers = [
        SimpleNodeParser.from_defaults(chunk_size=c) for c in sub_chunk_sizes
    ]
    all_nodes = []
    for base_node in base_nodes:
        for n in sub_node_parsers:
            sub_nodes = n.get_nodes_from_documents([base_node], show_progress=True)
            sub_inodes = [
                IndexNode.from_text_node(sn, base_node.node_id) for sn in sub_nodes
            ]
            all_nodes.extend(sub_inodes)

        # also add original node to node
        original_node = IndexNode.from_text_node(base_node, base_node.node_id)
        all_nodes.append(original_node)
    return all_nodes
