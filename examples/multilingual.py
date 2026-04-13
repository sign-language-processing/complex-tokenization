from complex_tokenization.draw import draw_dot_content
from complex_tokenization.graphs.units import register_script, utf8_clusters
from complex_tokenization.graphs.words import words
from complex_tokenization.languages.chinese.graph import chinese_character_to_graph
from complex_tokenization.languages.hebrew.decompose import decompose_cluster

register_script("Hebrew", decompose_cluster)
register_script("Han", chinese_character_to_graph)

text = " ".join([
    "hello",            # English
    "Fu\u0308r",        # German: "Für" with decomposed umlaut
    "\u05D1\u05BC\u05B0\u05E8\u05B5\u05D0\u05E9\u05C1\u05B4\u05BC\u05D9\u05EA",  # Hebrew: בְּרֵאשִׁ֖ית (Bereshit) with diacritics + cantillation
    "\u6797",           # Chinese: 林 (forest)
])

graph = words(text, connected=True, units=utf8_clusters)
img = draw_dot_content("\n".join(graph.dot()))
img.show()
