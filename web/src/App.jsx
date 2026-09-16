import { useEffect, useState } from "react";
import { api } from "./api.js";
import CustomerList from "./components/CustomerList.jsx";
import ProductCard from "./components/ProductCard.jsx";
import Section from "./components/Section.jsx";

const METRICS = [
  ["MAP@12", "map@12"],
  ["Hit rate@12", "hitrate@12"],
  ["Recall@100", "recall@100"],
  ["NDCG@12", "ndcg@12"],
];

export default function App() {
  const [metrics, setMetrics] = useState(null);
  const [customers, setCustomers] = useState([]);
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState(null);
  const [detail, setDetail] = useState(null);

  useEffect(() => {
    api.metrics().then(setMetrics).catch(() => setMetrics(null));
  }, []);

  useEffect(() => {
    api.customers(40, query).then((list) => {
      setCustomers(list);
      setSelected((current) => current ?? list[0]?.customer_id ?? null);
    });
  }, [query]);

  useEffect(() => {
    if (selected) api.customer(selected).then(setDetail);
  }, [selected]);

  return (
    <div className="shell">
      <header className="masthead">
        <div>
          <h1 className="wordmark">Aurora</h1>
          <p className="tagline">
            Twelve articles picked for each customer by a two-stage retrieval and ranking model, shown next to what
            they actually bought in the held-out week.
          </p>
        </div>
        <dl className="metrics">
          {metrics &&
            METRICS.map(([label, key]) => (
              <div className="metric" key={key}>
                <dt>{label}</dt>
                <dd>{metrics[key].toFixed(4)}</dd>
              </div>
            ))}
        </dl>
      </header>

      <div className="layout">
        <CustomerList
          customers={customers}
          selected={selected}
          query={query}
          onQuery={setQuery}
          onSelect={setSelected}
        />

        <main className="stage">
          {!detail ? (
            <div className="panel empty">Select a customer to see their recommendations.</div>
          ) : (
            <>
              <Section
                title="Recommended"
                description="The model's top twelve, in rank order. Green marks an article this customer went on to buy."
                badge={
                  <span className={`badge${detail.hits > 0 ? " scored" : ""}`}>
                    {detail.hits} of 12 bought
                  </span>
                }
              >
                <div className="grid">
                  {detail.recommendations.map((item) => (
                    <ProductCard key={item.article_id} item={item} showRank />
                  ))}
                </div>
              </Section>

              {detail.history.length > 0 && (
                <Section
                  title="What they bought before"
                  description="Their most recent purchases — the history the model had to work from."
                >
                  <div className="rail">
                    {detail.history.map((item) => (
                      <ProductCard key={item.article_id} item={item} />
                    ))}
                  </div>
                </Section>
              )}

              <Section
                title="What they actually bought"
                description="Purchases from the held-out week. Green marks the ones the model put in its top twelve."
              >
                <div className="rail">
                  {detail.purchases.map((item) => (
                    <ProductCard key={item.article_id} item={item} />
                  ))}
                </div>
              </Section>

              <p className="note">
                Customer <code>{detail.customer_id.slice(0, 16)}…</code> · most customers get no hits at all, which is
                what a hit rate of {metrics ? (metrics["hitrate@12"] * 100).toFixed(0) : "12"}% means. The list is
                ordered so the clearest examples come first.
              </p>
            </>
          )}
        </main>
      </div>
    </div>
  );
}
