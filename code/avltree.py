import ipdb
class AVLNode:
    parent = None
    lft = None
    rgt = None
    lvl = 1 # We've added a level...
    def __init__(self, key, val):
        self.key = key
        self.val = val

def search(node, key):
    if node is None: raise KeyError             # Empty leaf: It's not here
    if node.key == key: return node             # Found key: Return val
    elif key < node.key:                        # Less than the key?
        return search(node.lft, key)            # Go left
    else:                                       # Otherwise...
        return search(node.rgt, key)            # Go right

def avl_height(node):
        if node is None: return 0
        return node.lvl

def avl_balance(node):
    if node is None: return 0
    return avl_height(node.lft) - avl_height(node.rgt)

def left_rotate(node):
    parent = node.parent
    rgt = node.rgt
    T2 = rgt.lft
    # Perform rotation
    rgt.lft = node
    node.rgt = T2
    if T2 is not None: T2.parent = node
    rgt.parent = parent
    node.parent = rgt
    # Update heights
    node.lvl = 1 + max(avl_height(node.lft), avl_height(node.rgt))
    rgt.lvl = 1 + max(avl_height(rgt.lft), avl_height(rgt.rgt))
    # Return the new root
    return rgt


def right_rotate(node):
    parent = node.parent
    lft = node.lft
    T3 = lft.rgt
    # Perform rotation
    lft.rgt = node
    node.lft = T3
    if T3 is not None: T3.parent = node
    lft.parent = parent
    node.parent = lft
    # Update heights
    node.lvl = 1 + max(avl_height(node.lft), avl_height(node.rgt))
    lft.lvl = 1 + max(avl_height(lft.lft), avl_height(lft.rgt))
    # Return the new root
    return lft

def avl_rebalance_ins(node, key):
    # Step 3 - Get the balance factor
    balance = avl_balance(node)

    # Step 4 - If the node is unbalanced,
    # then try out the 4 cases
    # Case 1 - Left Left
    if balance > 1 and key < node.lft.key:
        return right_rotate(node)
    # Case 2 - Right Right
    if balance < -1 and key > node.rgt.key:
        return left_rotate(node)
    # Case 3 - Left Right
    if balance > 1 and key > node.lft.key:
        node.lft = left_rotate(node.lft)
        #node.lft.parent = node
        return right_rotate(node)
    # Case 4 - Right Left
    if balance < -1 and key < node.rgt.key:
        node.rgt = right_rotate(node.rgt)
        #node.rgt.parent = node
        return left_rotate(node)
    return node

def avl_rebalance_del(node):
    # Step 3 - Get the balance factor
    balance = avl_balance(node)

    # Step 4 - If the node is unbalanced,
    # then try out the 4 cases
    # Case 1 - Left Left
    if balance > 1 and avl_balance(node.lft) >= 0:
        return right_rotate(node)
    # Case 2 - Right Right
    if balance < -1 and avl_balance(node.rgt) <= 0:
        return left_rotate(node)
    # Case 3 - Left Right
    if balance > 1 and avl_balance(node.lft) < 0:
        node.lft = left_rotate(node.lft)
        node.lft.parent = node
        return right_rotate(node)
    # Case 4 - Right Left
    if balance < -1 and avl_balance(node.rgt) > 0:
        node.rgt = right_rotate(node.rgt)
        node.rgt.parent = node
        return left_rotate(node)

    return node


def avl_insert(node, key, val):
    # Step 1 - Perform normal BST
    if node is None: return AVLNode(key, val)
    if node.key == key: node.val = val
    elif key < node.key:
        node.lft = avl_insert(node.lft, key, val)
        node.lft.parent = node
    else:
        node.rgt = avl_insert(node.rgt, key, val)
        node.rgt.parent = node
    # Step 2 - Update the height of the
    # ancestor node
    node.lvl = 1 + max(avl_height(node.lft), avl_height(node.rgt))

    return avl_rebalance_ins(node, key)

def tree_min(node):
    while node.lft is not None:
        node = node.lft
    return node

def tree_max(node):
    while node.rgt is not None:
        node = node.rgt
    return node

def tree_transplant(tree, u, v):
    if u.parent is None:
        tree.root = v
    elif u is u.parent.lft:
        u.parent.lft = v
    else:
        u.parent.rgt = v
    if v is not None:
        v.parent = u.parent

def _tree_delete(tree, node):
    if node.lft is None:
        tree_transplant(tree, node, node.rgt)
        return node.rgt
    elif node.rgt is None:
        tree_transplant(tree, node, node.lft)
        return node.lft
    else:
        y = tree_min(node.rgt)
        if y is not node.rgt:
            tree_transplant(tree, y, y.rgt)
            y.rgt = node.rgt
            y.rgt.parent = y
        tree_transplant(tree, node, y)
        y.lft = node.lft
        y.lft.parent = y
    return y

def avl_delete_node(tree, root, key):
        # Find the node to be deleted and remove it
    if root is None:
        return root
    elif key < root.key:
        root.lft = avl_delete_node(tree, root.lft, key)
    elif key > root.key:
        root.rgt = avl_delete_node(tree, root.rgt, key)
    else:
        root = _tree_delete(tree, root)
    if root is None:
        return root

    root.lvl = 1 + max(avl_height(root.lft), avl_height(root.rgt))
    return avl_rebalance_del(root)

        # # Balance the tree
        # if balanceFactor > 1:
        #     if avl_balance(root.lft) >= 0:
        #         return right_rotate(root)
        #     else:
        #         root.lft = left_rotate(root.lft)
        #         return right_rotate(root)
        # if balanceFactor < -1:
        #     if avl_balance(root.rgt) <= 0:
        #         return left_rotate(root)
        #     else:
        #         root.rgt = right_rotate(root.rgt)
        #         return left_rotate(root)


def avl_delete(tree, node):
    # Step 1 - Perform normal BST

    node = _tree_delete(tree, node)

    # Step 2 - Update the height of the
    # ancestor node
    if node is not None:
        node.lvl = 1 + max(avl_height(node.lft), avl_height(node.rgt))

    return avl_rebalance(node, key)

class AVLTree:                                     # Simple wrapper
    root = None
    def __setitem__(self, key, val):
        self.root = avl_insert(self.root, key, val)
    def __getitem__(self, key):
        return search(self.root, key)
    def __delitem__(self, key):
        self.root = avl_delete_node(self, self.root, key)
    def __contains__(self, key):
        try: search(self.root, key)
        except KeyError: return False
        return True


def print2DUtil(node, space, COUNT=10):
    if node is None:
        return

    space += COUNT
    print2DUtil(node.rgt, space, COUNT=COUNT)
    print()
    for i in range(COUNT, space):
        print(end=" ")
    print(node.key)
    print2DUtil(node.lft, space, COUNT=COUNT)

def print_tree(node, COUNT=10):
    print2DUtil(node, 0, COUNT=COUNT)


preamble=r"""\documentclass[border=5pt,tikz]{standalone}
\usepackage{tikz}
\usetikzlibrary{graphs,graphdrawing}
\usegdlibrary{trees}
\usepackage{forest}
\usepackage{anyfontsize}

\begin{document}
\Large
\begin{forest}
  for tree={circle,draw, edge=->,
            minimum size=1.5em, % <-- added
            inner sep=1pt}
"""

outro=r"""\end{forest}
\end{document}
%%% Local Variables:
%%% mode: latex
%%% TeX-engine: luatex
%%% TeX-PDF-mode: t
%%% TeX-master: t
%%% TeX-master: t
%%% End:
"""
def tikz_tree(node, labels=False):
    S = []
    def myiter(nd):
        if nd is not None:
            S.append('[')
            if labels: S.append(str(nd.key))
            myiter(nd.lft)
            #if labels: S.append(str(nd.key))
            myiter(nd.rgt)
            S.append(']')
        else:
            S.append('[-1,phantom]')
    myiter(node)
    return preamble+''.join(S)+outro

def tree2texfile(atree, filename='/tmp/tree.tex', labels=False):
    with open(filename, 'w') as f:
        f.write(tikz_tree(atree.root, labels=labels))

# orig
# [5[1[0[-1,phantom][-1,phantom]][3[2[-1,phantom][-1,phantom]][4[-1,phantom][-1,phantom]]]][8[7[6[-1,phantom][-1,phantom]][-1,phantom]][9[-1,phantom][-1,phantom]]]]
# 6,8
# [5[1[0[-1,phantom][-1,phantom]][3[2[-1,phantom][-1,phantom]][4[-1,phantom][-1,phantom]]]][9[7[-1,phantom][-1,phantom]][-1,phantom]]]        
# 8,6
# [5[1[0[-1,phantom][-1,phantom]][3[2[-1,phantom][-1,phantom]][4[-1,phantom][-1,phantom]]]][7[-1,phantom][9[-1,phantom][-1,phantom]]]]
