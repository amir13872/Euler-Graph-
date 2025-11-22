import itertools
import os
import networkx as nx
import matplotlib.pyplot as plt

MAX_VERTICES = 6

def generate_all_graphs(n):
    nodes = list(range(n))
    edges = list(itertools.combinations(nodes, 2))
    total = 2 ** len(edges)

    print(f"\n✅ در حال ساخت {total} گراف ممکن ...")

    for mask in range(total):
        G = nx.Graph()
        G.add_nodes_from(nodes)

        for i, e in enumerate(edges):
            if (mask >> i) & 1:
                G.add_edge(*e)

        yield G

def is_new_graph(G, existing):
    for H in existing:
        if nx.is_isomorphic(G, H):
            return False
    return True

def analyze_graph(G):
    degrees = dict(G.degree())
    is_conn = nx.is_connected(G) if G.number_of_edges() > 0 else False
    is_euler = nx.is_eulerian(G)
    odd_deg = [v for v, d in degrees.items() if d % 2 == 1]
    is_semi = (len(odd_deg) == 2 and is_conn)

    cycles = nx.cycle_basis(G)

    # simple greedy dominating set
    remaining = set(G.nodes())
    dom = set()
    while remaining:
        best = max(G.nodes(), key=lambda v: len(set([v]) | set(G.neighbors(v)) & remaining))
        dom.add(best)
        remaining -= (set([best]) | set(G.neighbors(best)))

    return {
        "vertices": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "degrees": degrees,
        "connected": is_conn,
        "eulerian": is_euler,
        "semi_eulerian": is_semi,
        "cycles": cycles,
        "dominating_set_size": len(dom)
    }

def draw_graph(G, idx, outdir):
    plt.figure(figsize=(3,3))
    pos = nx.spring_layout(G, seed=idx)
    nx.draw(G, pos, with_labels=True, node_size=500)
    plt.title(f"Graph {idx}")
    plt.tight_layout()
    path = os.path.join(outdir, f"graph_{idx}.png")
    plt.savefig(path, dpi=140)
    plt.close()
    return path

def main():
    print("\n=== برنامه تولید گراف‌های غیر یکریخت ===")

    n = int(input("\n🔸 تعداد رئوس را وارد کنید (حداکثر 6): "))
    if n < 1 or n > MAX_VERTICES:
        print("❌ مقدار نامعتبر — باید بین 1 تا 6 باشد.")
        return

    outdir = "output_graphs"
    os.makedirs(outdir, exist_ok=True)

    unique_graphs = []

    for G in generate_all_graphs(n):
        if is_new_graph(G, unique_graphs):
            unique_graphs.append(G)

    print(f"\n✅ تعداد گراف‌های غیر یکریخت: {len(unique_graphs)}")

    info_path = os.path.join(outdir, "analysis.txt")
    with open(info_path, "w", encoding="utf-8") as f:

        for idx, G in enumerate(unique_graphs, start=1):
            analysis = analyze_graph(G)
            img = draw_graph(G, idx, outdir)

            f.write(f"\n--- Graph {idx} ---\n")
            for k, v in analysis.items():
                f.write(f"{k}: {v}\n")
            f.write(f"image: {img}\n")

    print(f"\n🎯 پایان کار!")
    print(f"📁 خروجی‌ها در پوشه:  {outdir}")
    print(f"📄 اطلاعات تحلیلی:   {info_path}\n")

if __name__ == "__main__":
    main()
