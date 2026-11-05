"""Explicit unresolved semantic candidates; none of these proposals changes IDs."""
import csv
from pathlib import Path


def write_review(mapping, destination):
    lookup = {(p['source'], p['source_name']): p for p in mapping['provenance']}
    a, u, v, s = mapping['sources']
    candidates = [
        ((a, 'Banh cuon'), (u, 'Steamed rice roll')),
        ((a, 'Banh cuon'), (s, 'Steamed Rice Rolls')),
        ((a, 'Banh cuon'), (v, 'banh-uot')),
        ((a, 'Banh gio'), (s, 'Vietnamese pyramid rice dumpling')),
        ((a, 'Banh mi'), (s, 'Grilled Pork Banh Mi')),
        ((a, 'Bun bo Hue'), (u, 'Hue beef rice vermicelli soup')),
        ((a, 'Bun bo Hue'), (s, 'Hue-style Spicy Beef Noodle Soup')),
        ((a, 'Bun thit nuong'), (s, 'Grilled Pork with Vermicelli')),
        ((a, 'Bun thit nuong'), (s, 'Grilled Pork Vermicelli')),
        ((s, 'Grilled Pork with Vermicelli'), (s, 'Grilled Pork Vermicelli')),
        ((a, 'Bun dau mam tom'), (s, 'Vermicelli with Tofu')),
        ((a, 'Chao long'), (s, 'Pork Offal Porridge')),
        ((a, 'Com tam'), (s, 'Broken Rice')),
        ((a, 'Pho'), (s, 'Beef Pho')),
        ((a, 'Goi cuon'), (s, 'Spring Rolls')),
        ((s, 'Spring Rolls'), (u, 'Fried spring rolls')),
        ((a, 'Xoi xeo'), (s, 'Sticky Rice')),
        ((a, 'Banh beo'), (u, 'Small steamed savory rice pancake')),
        ((a, 'Banh xeo'), (u, 'Coconut milk-flavored crepes with shrimp and beef')),
        ((v, 'banh-u'), (s, 'Vietnamese pyramid rice dumpling')),
    ]
    pairs = [(lookup[left], lookup[right]) for left, right in candidates]
    omelets = [p for p in mapping['provenance'] if p['source'] == u and p['source_name'] == 'omelet']
    if len(omelets) == 2: pairs.append(tuple(omelets))
    rows = []
    for left, right in pairs:
        assert left['canonical_id'] != right['canonical_id']
        row = {f'{side}_{field}': item[field] for side, item in [('left', left), ('right', right)]
               for field in ('source', 'source_class_id', 'source_name', 'canonical_id')}
        row.update(status='keep_separate_pending_review', evidence='Source names suggest a translation, variant, or ambiguous duplicate; class scope is not verified.')
        rows.append(row)
    with Path(destination).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    return {'review_pairs': len(rows),
            'review_canonical_classes': len({p['canonical_id'] for pair in pairs for p in pair}),
            'review_source_classes': len({(p['source'], p['source_class_id']) for pair in pairs for p in pair})}
