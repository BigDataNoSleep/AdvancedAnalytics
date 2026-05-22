# Cypher Queries — Epstein Graph Analysis

Companion reference to the report *Graph Analytics on the Epstein Document Network*.
All queries below were executed against a Memgraph 3.10.1 instance with the
Epstein-doc-explorer snapshot loaded (546,648 nodes, 1,432,277 edges).

Queries are grouped into three phases:

- **A. Exploration** — one-off inventory queries to understand the graph
- **B. Subgraph 1 — Flight co-travel network** → produces **Figure 1** in the report
- **C. Subgraph 2 — Bipartite person-location network** → produces **Figure 2**
---

## A. Exploration queries

These were used to understand the schema, vocabulary, and entity quality of the
graph before committing to a filter design.

### A.1 — Action vocabulary inventory

What action strings exist on `RELATED_TO` edges that look flight-related? This
drives the filter choice in B.1.

```cypher
MATCH ()-[r:RELATED_TO]->()
WHERE toLower(r.action) CONTAINS 'flew'
   OR toLower(r.action) CONTAINS 'flight'
   OR toLower(r.action) CONTAINS 'travel'
   OR toLower(r.action) CONTAINS 'board'
   OR toLower(r.action) CONTAINS 'plane'
   OR toLower(r.action) CONTAINS 'jet'
   OR toLower(r.action) CONTAINS 'aircraft'
   OR toLower(r.action) CONTAINS 'pilot'
RETURN r.action AS action, count(*) AS n
ORDER BY n DESC
LIMIT 50;
```

Key finding: the strict co-travel actions are `flew with` (30), `flew on` (25),
`traveled with` (99). `traveled to` (299) is destination-oriented and excluded.
`serves on board of` and `served on board of` are false positives (corporate
boards, not aircraft) and excluded.

### A.2 — Travel-related tag inventory

```cypher
MATCH (t:Tag)
WHERE toLower(t.name) CONTAINS 'travel'
   OR toLower(t.name) CONTAINS 'flight'
   OR toLower(t.name) CONTAINS 'aviation'
   OR toLower(t.name) CONTAINS 'air'
   OR toLower(t.name) CONTAINS 'jet'
RETURN t.name AS tag
LIMIT 50;
```

Tag-based filtering was explored but action-based filtering was preferred for
the final flight subgraph (cleaner and more direct).

### A.3 — Schema verification

Property inventory for the principal node and edge types.

```cypher
MATCH (c:Claim) RETURN keys(c) AS properties LIMIT 1;
MATCH (d:Document) RETURN keys(d) AS properties LIMIT 1;
MATCH (a:Entity)-[r:RELATED_TO]->(b:Entity) RETURN keys(r) AS properties LIMIT 1;
```

Outputs:
- `Claim`: `action`, `doc_id`, `claim_id`, `sequence_order` *(no timestamps directly)*
- `Document`: `date_range_earliest`, `date_range_latest`, `category`, plus summary/text fields
- `RELATED_TO`: `action`, `doc_id`, `claim_id`, `sequence_order`

### A.4 — Location schema and edge connectivity

```cypher
MATCH (l:Location)
RETURN l.name AS location, keys(l) AS properties
LIMIT 10;

MATCH (l:Location)<-[r]-(n)
RETURN labels(n) AS source_label, type(r) AS edge_type, count(*) AS n
ORDER BY n DESC;
```

Result: `Location` nodes are connected only via `AT_LOCATION` edges from
`Claim` nodes (33,530 such edges total).

### A.5 — Top locations inventory

```cypher
MATCH (l:Location)<-[:AT_LOCATION]-(c:Claim)
WITH l, count(c) AS n_claims
WHERE n_claims >= 5
RETURN l.name AS location, n_claims
ORDER BY n_claims DESC
LIMIT 50;
```

Surfaced the location duplication problem (multiple variants per real place)
that drove the canonicalisation step in C.1.

### A.6 — Find canonical Epstein-related locations

Confirms which key destinations exist as `Location` nodes.

```cypher
MATCH (l:Location)
WHERE toLower(l.name) CONTAINS 'little st'
   OR toLower(l.name) CONTAINS 'james'
   OR toLower(l.name) CONTAINS 'zorro'
   OR toLower(l.name) CONTAINS 'palm beach'
   OR toLower(l.name) CONTAINS 'virgin island'
   OR toLower(l.name) CONTAINS 'epstein island'
   OR toLower(l.name) CONTAINS 'mar-a-lago'
RETURN l.name AS location;
```

Returned 568 matching `Location` nodes including all key destinations:
Little Saint James (several variants), Zorro Ranch, Mar-a-Lago, El Brillo Way,
multiple Palm Beach variants.

### A.7 — Entity dedup verification

Run per cluster to identify variant forms of the same person. Pattern:

```cypher
MATCH (e:Entity)-[r:RELATED_TO]-()
WHERE toLower(e.name) IN ['bill clinton','president clinton','clinton',
                          'william clinton','william jefferson clinton','wjc',
                          'president william clinton','former president clinton']
RETURN e.name AS variant, count(r) AS edges
ORDER BY edges DESC;
```

Repeated for Ghislaine Maxwell, Alan M. Dershowitz, and Prince Andrew variant
lists. Identified the entity duplicates documented in the report
but no entity-level MERGE was applied as the change was marginal (often only + 1)

### A.8 — Visitors to a specific location

Diagnostic query for the Little Saint James inner-circle finding.

```cypher
MATCH (l:Location)
WHERE toLower(l.name) CONTAINS 'little saint james'
   OR toLower(l.name) CONTAINS 'little st. james'
   OR toLower(l.name) CONTAINS 'little st james'
MATCH (e:Entity)<-[:ACTOR|TARGET]-(c:Claim)-[:AT_LOCATION]->(l)
RETURN e.name AS visitor, count(DISTINCT c) AS mentions, l.name AS variant
ORDER BY mentions DESC;
```

---

## B. Subgraph 1 — Flight co-travel network

### B.1 — Final flight subgraph filter and export

> **This is the query that produces Figure 1 in the report.**

Filters `RELATED_TO` edges to strict co-travel actions, excludes any endpoint
that is an aircraft or known location-as-Entity, then returns the graph
pattern for Gephi export.

```cypher
MATCH (a:Entity)-[r:RELATED_TO]->(b:Entity)
WHERE toLower(r.action) IN ['flew with','flew on','traveled with']
  AND NOT toLower(a.name) CONTAINS 'jet'
  AND NOT toLower(a.name) CONTAINS 'plane'
  AND NOT toLower(a.name) CONTAINS 'aircraft'
  AND NOT toLower(a.name) CONTAINS 'airline'
  AND NOT toLower(a.name) CONTAINS 'boeing'
  AND NOT toLower(a.name) CONTAINS 'gulfstream'
  AND NOT toLower(b.name) CONTAINS 'jet'
  AND NOT toLower(b.name) CONTAINS 'plane'
  AND NOT toLower(b.name) CONTAINS 'aircraft'
  AND NOT toLower(b.name) CONTAINS 'airline'
  AND NOT toLower(b.name) CONTAINS 'boeing'
  AND NOT toLower(b.name) CONTAINS 'gulfstream'
RETURN a, r, b;
```

---

## C. Subgraph 2 — Bipartite person-location network

### C.1 — Location canonicalisation (database-modifying)

Merges variants of the major location clusters into canonical nodes, summing
`VISITED` edge weights. (Run *before* C.2 if you want the canonical names on
the merged edges; or after, in which case re-run C.2 to refresh weights.)

```cypher
WITH [
  ['New York City', ['New York','New York, NY','Manhattan','NYC',
                     'Lower East Side, New York','New York, New York',
                     'New York mansion','Manhattan mansion','Manhattan residence',
                     'New York NY','9 E. 71st Street, Manhattan']],
  ['Palm Beach',    ['Palm Beach, Florida','Palm Beach, FL',
                     'Town of Palm Beach','Palm Beach County',
                     'Palm Beach County, Florida','West Palm Beach',
                     'West Palm Beach, Florida','West Palm Beach, FL',
                     'El Brillo Way','El Brillo Way, Palm Beach',
                     '358 El Brillo Way, Palm Beach',
                     '358 El Brillo Way, Palm Beach, FL',
                     '358 El Brillo Way, Palm Beach, Florida',
                     '358 El Brillo Way, Palm Beach, FL 33480',
                     "Epstein's home","Epstein's house",
                     "Jeffrey Epstein's residence",
                     "Jeffrey Epstein's house",
                     'Palm Beach mansion','Palm Beach house',
                     'house on El Brillo Way']],
  ['Little Saint James', ['Little St. James','Little St James',
                          'Little Saint James Island','Little St. James Island',
                          'Little St. James island','Little St. James, Caribbean',
                          "Epstein's island","Epstein's island, US Virgin Islands"]],
  ['Mar-a-Lago',    ['Mar-a-Lago Club','Mar-a-Lago Inc.','Mar-a-Lago Club, Inc.',
                     'Mar-a-Lago reception','Mar a Lago']],
  ['U.S. Virgin Islands', ['Virgin Islands','USVI','US Virgin Islands',
                           'St. Thomas',
                           'Carlton, St. Thomas, U.S. Virgin Islands',
                           'Carlton, St. Thomas, US Virgin Islands']],
  ['Washington, D.C.', ['Washington','Washington D.C.','Washington DC']]
] AS mapping

UNWIND mapping AS pair
WITH pair[0] AS canonical_name, pair[1] AS variant_names

MATCH (canonical:Location {name: canonical_name})
MATCH (variant:Location)
WHERE variant.name IN variant_names AND variant <> canonical
MATCH (variant)<-[v:VISITED]-(e:Entity)
WITH canonical, variant, e, v.weight AS old_weight, v, variant_names
DELETE v
WITH canonical, variant, e, old_weight, variant_names
MERGE (e)-[new_v:VISITED]->(canonical)
ON CREATE SET new_v.weight = old_weight
ON MATCH SET new_v.weight = new_v.weight + old_weight
WITH DISTINCT variant_names

MATCH (l:Location)
WHERE l.name IN variant_names
DETACH DELETE l;
```

### C.2 — Derive VISITED edges (database-modifying)

Creates a new directed edge type `(:Entity)-[:VISITED]->(:Location)` connecting
every flight-network member to every location they're documented at, with
weight = number of distinct supporting claims, filtered to weight ≥ 2.

```cypher
MATCH (a:Entity)-[r:RELATED_TO]-(b:Entity)
WHERE toLower(r.action) IN ['flew with','flew on','traveled with']
  AND NOT toLower(a.name) CONTAINS 'jet'
  AND NOT toLower(a.name) CONTAINS 'plane'
  AND NOT toLower(a.name) CONTAINS 'aircraft'
  AND NOT toLower(a.name) CONTAINS 'airline'
  AND NOT toLower(a.name) CONTAINS 'boeing'
  AND NOT toLower(a.name) CONTAINS 'gulfstream'
  AND NOT toLower(b.name) CONTAINS 'jet'
  AND NOT toLower(b.name) CONTAINS 'plane'
  AND NOT toLower(b.name) CONTAINS 'aircraft'
  AND NOT toLower(b.name) CONTAINS 'airline'
  AND NOT toLower(b.name) CONTAINS 'boeing'
  AND NOT toLower(b.name) CONTAINS 'gulfstream'
WITH collect(DISTINCT a) + collect(DISTINCT b) AS flight_people
UNWIND flight_people AS e
WITH DISTINCT e

MATCH (e)<-[:ACTOR|TARGET]-(c:Claim)-[:AT_LOCATION]->(l:Location)
WHERE NOT toLower(l.name) IN ['unspecified','email','united states','europe',
                              'florida','washington']
WITH e, l, count(DISTINCT c) AS weight
WHERE weight >= 2
MERGE (e)-[v:VISITED]->(l)
SET v.weight = weight;
```

To undo:
```cypher
MATCH ()-[v:VISITED]->() DELETE v;
```

### C.3 — Bipartite export query


After C.1 and C.2 have been run, the bipartite graph is one MATCH away:

```cypher
MATCH (e:Entity)-[v:VISITED]->(l:Location)
RETURN e, v, l;
```

### C.4 — Verification after canonicalisation

Sanity check that the merge produced the expected per-canonical totals:

```cypher
MATCH (l:Location)
WHERE l.name IN ['New York City','Palm Beach','Little Saint James',
                 'Mar-a-Lago','U.S. Virgin Islands','Washington, D.C.']
OPTIONAL MATCH (l)<-[v:VISITED]-(e:Entity)
RETURN l.name AS canonical,
       count(DISTINCT e) AS unique_visitors,
       sum(v.weight) AS total_weight
ORDER BY total_weight DESC;
```

Returns the six canonical locations with their merged visitor counts and
summed weights.

---

### queries voor geld transacties 

# eerste versie graph die geld related is 
MATCH (a:Entity)-[r:RELATED_TO]->(b:Entity)
WHERE r.action CONTAINS "paid" 
   OR r.action CONTAINS "$" 
   OR r.action CONTAINS "money"
   OR r.action CONTAINS "transfer"
   OR b.name CONTAINS "$"
RETURN a, r, b;


# overzicht clusters (uiteindelijk niet gebruikt)
MATCH (c:Cluster)<-[:IN_CLUSTER]-(cl:Claim)
WITH c, count(cl) AS Aantal_Claims, collect(cl.action)[0..5] AS Voorbeelden
RETURN c.cluster_id AS Cluster_ID, Aantal_Claims, Voorbeelden
ORDER BY Aantal_Claims DESC;

## Meest complete geld query tot nu toe: volledige geld graph 
MATCH (a:Entity)-[r:RELATED_TO]->(b:Entity)
WHERE (
    toLower(r.action) =~ ".*(pay|paid|transfer|wire|fund|settle|loan|cheque|cash|dollar|\\$|euro/|€|pound/|£|donat).*"
    
    OR a.name CONTAINS "$" OR a.name CONTAINS "€" OR a.name CONTAINS "£"
    OR b.name CONTAINS "$" OR b.name CONTAINS "€" OR b.name CONTAINS "£"
)
RETURN a, r, b;

# zien hoeveel edges elke node heeft in de geld graph (dus hoeveel transacties elke relevante speler hier)
MATCH (a:Entity)-[r:RELATED_TO]->(b:Entity)
WHERE (
    toLower(r.action) =~ ".*(pay|paid|transfer|wire|fund|settle|loan|cheque|cash|dollar|\\$|euro/|€|pound/|£|donat).*"
    
    OR a.name CONTAINS "$" OR a.name CONTAINS "€" OR a.name CONTAINS "£"
    OR b.name CONTAINS "$" OR b.name CONTAINS "€" OR b.name CONTAINS "£"
)
WITH a, count(r) AS uitgaande_transacties
RETURN a.name AS Entiteit, uitgaande_transacties
ORDER BY uitgaande_transacties DESC
LIMIT 10;

### graph, enkel mensen die 5 of meer transacties hebben gedaan 
MATCH (a:Entity)-[r:RELATED_TO]->(b:Entity)
WHERE (
    toLower(r.action) =~ ".*(pay|paid|transfer|money|wire|fund|settle|loan|cheque|cash|dollar|\\$|euro/|€|pound/|£|donat).*"
    
    OR a.name CONTAINS "$" OR a.name CONTAINS "€" OR a.name CONTAINS "£"
    OR b.name CONTAINS "$" OR b.name CONTAINS "€" OR b.name CONTAINS "£"
)
WITH a, b, count(r) AS aantal
WHERE aantal > 4
MERGE (a)-[g:GELDSTROOM]->(b)
SET g.transacties = aantal
RETURN a, g, b;
## tabel met de exacte transfers van de mensen die er 5 of meer gedaan hebben
MATCH (a:Entity)-[r:RELATED_TO]->(b:Entity)
WHERE (
    toLower(r.action) =~ ".*(pay|paid|transfer|money|wire|fund|settle|loan|cheque|cash|dollar|\\$|euro|€|pound|£|donat).*"
    OR a.name CONTAINS "$" OR b.name CONTAINS "$"
)
WITH a, b, collect(r) AS relaties
WHERE size(relaties) >= 5
UNWIND relaties AS r
RETURN a.name AS Van, b.name AS Naar, r.action AS Exacte_Actie, r.doc_id AS Document_ID
ORDER BY Van ASC, Naar ASC;

