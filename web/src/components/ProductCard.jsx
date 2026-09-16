import { useState } from "react";

export default function ProductCard({ item, showRank = false }) {
  const [broken, setBroken] = useState(false);

  return (
    <article className={`card${item.hit ? " is-hit" : ""}`} title={`${item.name} — ${item.product_type}`}>
      <figure>
        {broken ? (
          <span className="missing">no photo</span>
        ) : (
          <img src={item.image} alt={item.name} loading="lazy" onError={() => setBroken(true)} />
        )}
      </figure>
      {showRank && <span className="rank">#{item.rank}</span>}
      {item.hit && <span className="tick" aria-label="bought">✓</span>}
      <figcaption>
        <p className="name">{item.name}</p>
        <p className="meta">
          {item.product_type} · {item.colour}
        </p>
      </figcaption>
    </article>
  );
}
