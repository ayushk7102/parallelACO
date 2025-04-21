graph [
  directed 0
  node [
    id 0
    label "A"
  ]
  node [
    id 1
    label "B"
  ]
  node [
    id 2
    label "C"
  ]
  node [
    id 3
    label "D"
  ]
  node [
    id 4
    label "E"
  ]
  node [
    id 5
    label "F"
  ]
  node [
    id 6
    label "G"
  ]
  node [
    id 7
    label "H"
  ]
  node [
    id 8
    label "I"
  ]
  node [
    id 9
    label "J"
  ]
  # Community 1: nodes 0-3 (A-D)
  edge [
    source 0
    target 1
  ]
  edge [
    source 0
    target 2
  ]
  edge [
    source 0
    target 3
  ]
  edge [
    source 1
    target 2
  ]
  edge [
    source 2
    target 3
  ]
  # Community 2: nodes 4-6 (E-G)
  edge [
    source 4
    target 5
  ]
  edge [
    source 4
    target 6
  ]
  edge [
    source 5
    target 6
  ]
  # Community 3: nodes 7-9 (H-J)
  edge [
    source 7
    target 8
  ]
  edge [
    source 7
    target 9
  ]
  edge [
    source 8
    target 9
  ]
  # Inter-community edges
  edge [
    source 3
    target 4
  ]
  edge [
    source 6
    target 7
  ]
]