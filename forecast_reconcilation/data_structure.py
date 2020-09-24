
from collections import OrderedDict

class HierarchyTree(object):

    def __init__(self):
        self._nodes = OrderedDict()
        self._bottom = []
    
    def add(self, node):
        self._nodes[node.name] = node
    
    def remove(self, node):
        del self._nodes[node.name]
        for parent in node.parent:
            parent.extend(node.children)
            self._nodes[parent.name] = parent 
        for child in node.children:
            child.parent.remove(node)
            child.parent.extend(node.parent)
            self._nodes[child.name] = child

    def isin(self, node):
        return node.name in self._nodes

    @property
    def nodes(self):
        return self._nodes
    
    @property
    def num_nodes(self):
        return len(self._nodes)
    
    @property 
    def root(self):
        return self._nodes["root"]
    
    def get_node(self, nodename):
        return self._nodes[nodename]
    
    @property
    def num_bottom_level(self):
        return len(self._bottom)
    
    def add_bottom(self, node):
        self._bottom.append(node)
    
    @property
    def bottom(self):
        return self._bottom
    
    @staticmethod
    def from_nodes(hierarchy: dict):
        tree = HierarchyTree()
        queue = ["root"] 
        node = TreeNode("root")
        tree.add(node)
        while queue:
            nodename = queue.pop(0)
            node = TreeNode(nodename)
            if tree.isin(node):
                node = tree.get_node(nodename)
            if nodename not in hierarchy:
                tree.add(node)
                tree.add_bottom(node)
                continue
            for child in hierarchy[nodename]:
                child = TreeNode(child)
                if not tree.isin(child):
                    child.parent.append(node)
                    node.children.append(child)
                    tree.add(child)
                else:
                    child = tree.get_node(child.name)
                    child.parent.append(node)
                    node.children.append(child)
                queue.append(child.name)
        return tree
    
    def __repr__(self):
        print([nodename for nodename in self._nodes])

class TreeNode(object):

    def __init__(self, name=None):
        self.name = name 
        self.children = []
        self.parent = []
    
    def append(self, child):
        self._children.append(child)
    
    def __repr__(self):
        print(self.name)
