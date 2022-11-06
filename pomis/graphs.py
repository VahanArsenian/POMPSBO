import networkx as nx
import typing as tp

import npsem.model


class Projection(nx.DiGraph):

    def __init__(self, graph: nx.DiGraph, confounded_triplets: tp.Set[tp.Tuple[str, str, str]],
                 confounding_variables: tp.Set[str]):
        super().__init__(graph)
        self.confounded_triplets = confounded_triplets
        self.confounding_variables = confounding_variables


class NonManGraph(nx.DiGraph):

    def __init__(self, non_man_variables: tp.Iterable[str] = (), **attr):
        super().__init__(**attr)
        self.non_man_variables = set(non_man_variables)

    def add_non_man_node(self, nodes: tp.Union[tp.Iterable[str], str]):
        if isinstance(nodes, str):
            nodes = [nodes]
        self.non_man_variables = set(list(self.non_man_variables) + nodes)

    @property
    def projection(self) -> Projection:
        non_man = self.non_man_variables
        man = set(self.nodes) - non_man

        projection = nx.DiGraph()
        projection.add_nodes_from(man)
        confounded_triplets = set()
        confounding_variables = set()

        for v1 in man:
            for v2 in man:
                if v1 == v2:
                    continue
                if self.has_edge(v1, v2):
                    projection.add_edge(v1, v2)
                    continue
                if nx.has_path(self, v1, v2):
                    for path in nx.all_simple_edge_paths(self, v1, v2):
                        path_nodes = set(sum(map(list, path), [])) - {v1, v2}
                        if len(path_nodes) > 0 and path_nodes.issubset(non_man):
                            projection.add_edge(v1, v2)

        for v1 in man:
            for v2 in man:
                for u in non_man:
                    if v1 == v2:
                        continue

                    c1 = False
                    c2 = False

                    for path in nx.all_simple_edge_paths(self, u, v1):
                        path_nodes = set(sum(map(list, path), []))
                        if (path_nodes - {v1}).issubset(non_man):
                            c1 = True
                            break
                    if c1:
                        for path in nx.all_simple_edge_paths(self, u, v2):
                            path_nodes = set(sum(map(list, path), []))
                            if (path_nodes - {v2}).issubset(non_man):
                                c2 = True
                                break

                    if c1 and c2:
                        projected_confounded_pair = sorted([v1, v2])
                        projected_latent_name = f"u-{projected_confounded_pair[0]}-{projected_confounded_pair[1]}"

                        confounding_variables |= {projected_latent_name}
                        confounded_triplets |= {(*projected_confounded_pair, projected_latent_name)}

                        projection.add_edge(projected_latent_name, v1)
                        projection.add_edge(projected_latent_name, v2)
        return Projection(projection, confounded_triplets, confounding_variables)


class CausalGraph(npsem.model.CausalDiagram):

    def __init__(self, projection: Projection):
        nodes = set(projection.nodes) - projection.confounding_variables
        directed = [(v1, v2) for v1, v2 in projection.edges if len({v1, v2} & projection.confounding_variables) == 0]
        bi_directed = [(v1, v2, f"u_{i}") for i, (v1, v2, _) in enumerate(projection.confounded_triplets)]
        super(CausalGraph, self).__init__(nodes, directed, bi_directed)

