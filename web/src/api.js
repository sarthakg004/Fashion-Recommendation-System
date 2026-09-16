class AuroraApi {
  constructor(base = "/api") {
    this.base = base;
  }

  async #get(path) {
    const response = await fetch(`${this.base}${path}`);
    if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
    return response.json();
  }

  metrics() {
    return this.#get("/metrics");
  }

  customers(limit = 40, query = "") {
    const params = new URLSearchParams({ limit });
    if (query) params.set("q", query);
    return this.#get(`/customers?${params}`);
  }

  customer(id) {
    return this.#get(`/customers/${id}`);
  }
}

export const api = new AuroraApi();
