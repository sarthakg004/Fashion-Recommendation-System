export default function CustomerList({ customers, selected, query, onQuery, onSelect }) {
  return (
    <aside className="panel sidebar">
      <h2>Customers</h2>
      <input
        className="search"
        placeholder="Filter by id…"
        value={query}
        onChange={(event) => onQuery(event.target.value)}
      />
      <div className="customer-list">
        {customers.map((customer) => (
          <button
            key={customer.customer_id}
            className="customer"
            aria-current={customer.customer_id === selected}
            onClick={() => onSelect(customer.customer_id)}
          >
            <code>{customer.customer_id.slice(0, 12)}…</code>
            <span className="count">
              {customer.hits > 0 ? `${customer.hits} hit${customer.hits > 1 ? "s" : ""}` : `${customer.bought} bought`}
            </span>
          </button>
        ))}
        {customers.length === 0 && <p className="note">No customer matches that prefix.</p>}
      </div>
    </aside>
  );
}
