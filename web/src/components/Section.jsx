export default function Section({ title, description, badge, children }) {
  return (
    <section className="panel section">
      <header>
        <div>
          <h3>{title}</h3>
          {description && <p>{description}</p>}
        </div>
        {badge}
      </header>
      {children}
    </section>
  );
}
