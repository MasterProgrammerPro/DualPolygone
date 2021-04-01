# TorusCurvePolygone

## Utilisation

Vous devez creer un numpy array qui contient des sommets de votre polygone et creer votre Torus en utilisant ça.

```python
vertices = np.array([[-2, 0], [-1, 1], [2, 0], [0, -1]])
quad = TorusCurvePolygone(vertices=vertices)
```
Vous pouvez aussi determiner votre resolution.
```python
vertices = np.array([[-2, 0], [-1, 1], [2, 0], [0, -1]])
quad = TorusCurvePolygone(vertices=vertices, resolution=128)
```

## Problemes
1) La fonction gamma_list_p() ne marche pas.
2) Probleme avec des resolution grande a cause de "maximum recursion depth".
