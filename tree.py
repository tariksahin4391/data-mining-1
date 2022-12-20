class TreeNode:
    id = None
    is_leaf = False
    left = None
    right = None
    parent = None
    data = None

    def __init__(self, left=None, right=None, parent=None, is_leaf=False, data=None, id='default'):
        self.left = left
        self.right = right
        self.data = data
        self.parent = parent
        self.is_leaf = is_leaf
        self.id = id


def print_all_nodes(node: TreeNode):
    print('node id is ', node.id)
    print('node data is ', node.data)
    if not node.left is None:
        print('get left side')
        print_all_nodes(node.left)
    if not node.right is None:
        print('get right side')
        print_all_nodes(node.right)


root = TreeNode(data=3, id='root')
left = TreeNode(data=5, parent=root, is_leaf=True, id='left')
right = TreeNode(data=8, parent=root, is_leaf=True, id='right')
root.left = left
root.right = right

print_all_nodes(root)
