import itertools

class AANode:
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

def skew(node):                                 # Basically a right rotation
    if node is None or node.lft is None: return node    # No need for a skew
    if node.lft.lvl != node.lvl: return node    # Still no need
    lft = node.lft                              # The 3 steps of the rotation
    node.lft = lft.rgt
    lft.rgt = node
    lft.parent = node
    node.parent = lft
    return lft                                  # Switch pointer from parent

def split(node):                                # Left rotation & level incr.
    if node is None or node.rgt is None or node.rgt.rgt is None: return node
    if node.rgt.rgt.lvl != node.lvl: return node
    rgt = node.rgt
    node.rgt = rgt.lft
    rgt.lft = node
    rgt.lvl += 1                                # This has moved up
    rgt.parent = node

    return rgt                                  # This should be pointed to

def aa_insert(node, key, val):
    if node is None: return AANode(key, val)
    if node.key == key: node.val = val
    elif key < node.key:
        node.lft = aa_insert(node.lft, key, val)
        node.lft.parent = node                  # and the parent
    else:
        node.rgt = aa_insert(node.rgt, key, val)
        node.rgt.parent = node                  # and the parent
    node = skew(node)                           # In case it's backward
    node = split(node)                          # In case it's overfull
    return node
def tree_min(node):
    while node.lft is not None:
        node = node.lft
    return node

def tree_max(node):
    while node.rgt is not None:
        node = node.rgt
    return node

def tree_successor(node):
    if node.rgt is not None:
        return tree_min(node.rgt)
    p = node.parent
    while p is not None and p.rgt is node:
        node = p
        p = node.parent
    return p

def tree_predecessor(node):
    if node.lft is not None:
        return tree_max(node.lft)
    p = node.parent
    while p is not None and p.lft is node:
        node = p
        p = node.parent
    return p

def decrease_level(root):
    should_be = min(0 if root.lft is None else root.lft.lvl, 0 if root.rgt is None else root.rgt.lvl) + 1
    if should_be < root.lvl:
        root.lvl = should_be
        if root.rgt is not None and should_be < root.rgt.lvl:
            root.rgt.lvl = should_be
    return root

def aa_remove (root, key):
    if root is None:
        return root
    elif key > root.key:
        root.rgt = aa_remove(root.rgt, key)
    elif key < root.key:
        root.lft = aa_remove(root.lft, key)
    else:
        if root.lft is None and root.rgt is None:
            return root.rgt
        elif root.lft is None:
            L = tree_successor(root)
            root.rgt = aa_remove(root.rgt, L.key)
            root.key = L.key
        else:
            L = tree_predecessor(root)
            root.lft = aa_remove(root.lft, L.key)
            root.key = L.key
    root = decrease_level(root)
    root = skew(root)
    root.rgt = skew(root.rgt)
    if root.rgt is not None:
        root.rgt.rgt = skew(root.rgt.rgt)
    root = split(root)
    root.rgt = split(root.rgt)
    return root

# def aa_remove (root, key):
#     if root is not None:
#         if root.key == key:
#             if root.lft is not None and root.rgt is not None:
#                 heir = root.lft
#                 while heir.rgt is not None:
#                     heir = heir.rgt
#                 root.key = heir.key
#                 root.val = heir.val
#                 root.lft = aa_remove(root.lft, root.key)
#             elif root.lft is None:
#                 root = root.rgt
#             else:
#                 root = root.lft
#         elif root.key < key:
#             root.rgt = aa_remove(root.rgt, key)
#         else:
#             root.lft = aa_remove(root.lft, key)
#     if (root.lft.lvl < (root.lvl - 1) or
#         root.rgt.lvl < (root.lvl - 1)):
#         root.lvl -= 1
#         if root.rgt.lvl > root.lvl:
#             root.rgt.lvl = root.lvl
#         root = split(skew(root))
#     return root

class AATree:                                     # Simple wrapper
    root = None
    def __setitem__(self, key, val):
        self.root = aa_insert(self.root, key, val)
    def __getitem__(self, key):
        return search(self.root, key)
    def __delitem__(self, key):
        self.root = aa_remove(self.root, key)
    def __contains__(self, key):
        try: search(self.root, key)
        except KeyError: return False
        return True

def hash_a_tree(node):
    S = []
    def myiter(nd):
        if nd is not None:
            S.append('(')
            myiter(nd.lft)
            # S.append(str(nd.key))
            myiter(nd.rgt)
            S.append(')')
        else:
            S.append(' ')
    myiter(node)
    return ''.join(S)


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


# [6[1[0[-1,phantom][-1,phantom]][3[2[-1,phantom][-1,phantom]][4[-1,phantom][5[-1,phantom][-1,phantom]]]]][8[7[-1,phantom][-1,phantom]][9[-1,phantom][-1,phantom]]]]
# 9, 3
# [2[0[-1,phantom][1[-1,phantom][-1,phantom]]][6[4[-1,phantom][5[-1,phantom][-1,phantom]]][7[-1,phantom][8[-1,phantom][-1,phantom]]]]]        
