# Image Download Guide

You need 70 images, one per entity, saved as `data/images/{entity_id}.jpg`.

**Budget:** ~55 minutes total. Work in batches.

---

## The rule

Every image must be named **exactly** `{entity_id}.jpg`. The IDs are listed in `data/entities.csv` (first column).

- `c_vader.jpg` ✅
- `Vader.jpg` ❌ (won't be found)
- `c_vader.png` ⚠️ (works, but use .jpg for consistency)

Size: ~500-800px on the long side is plenty.

---

## Sources

**For characters (28) → Wookieepedia**
1. Visit `https://starwars.fandom.com/wiki/{Character_Name}`
2. Right-click the main infobox image → "Save image as"
3. Rename to `{entity_id}.jpg`

**For ships (12) → Wookieepedia, or Google `[ship name] transparent png`**
Same as characters — main infobox is usually the cleanest.

**For planets (8) → Wookieepedia main infobox**
Either the orbit view or a surface shot. Pick whichever your `visual_description` describes. For Crait specifically grab a surface shot showing the red salt — orbit shot is boring.

**For episodes/arcs (7) → Google Images for the specific scene**
Pick a clean frame, not a poster (with one exception):
- `e_order66` → search "Order 66 Anakin Jedi Temple" → grab the temple-march frame
- `e_vader_hallway` → "Vader hallway scene Rogue One" → red-lit corridor
- `e_obi_vader_rematch` → "Obi-Wan vs Vader Kenobi show broken mask" → cracked helmet reveal
- `e_anakin_fall` → "Anakin Mustafar duel" → high-ground lava-bank moment
- `e_andor_s1` → official Andor S1 poster (this one is fine as a poster — Luthen + Cassian)
- `e_mando_s2_finale` → "Mandalorian Luke Skywalker reveal" → hooded Luke ramp moment
- `e_book_of_boba` → "Boba Fett throne Tatooine" → Boba on the stone throne

---

## Fast workflow (45 min)

Don't go entity-by-entity hunting around. Do all 57 in 4 batches:

**Batch 1 (~18 min): All 37 characters (including 4 droids and 3 newly added)**
Open Wookieepedia. Search → right-click infobox → save → rename. About 30 seconds per character.

**Batch 2 (~10 min): All 12 ships**
Same approach. Wookieepedia ship pages.

**Batch 3 (~13 min): All 11 planets (biome-diverse — Scarif, Kamino, Exegol newly added)**
Same approach. Take 5 seconds longer per planet to pick the right shot (orbit vs surface).

**Batch 4 (~14 min): All 10 episodes**
Slowest because each is a Google Images search. Pick clean frames. ~80 seconds each.

---

## Important: don't reuse images for similar entries

Three pairs of entries need *different* images:

1. **`c_obiwan_pt` vs `c_obiwan_ot`** — younger Ewan (prequel-era, brown hair) vs older Ewan (Kenobi-show, full grey beard, weathered). Don't use the same headshot.
2. **`c_palpatine_pt_ot` vs `c_palp_sequels`** — normal Sidious hood vs the corpse-like resurrected version from TROS (hooked into cables).
3. **`c_thrawn_legends`** — use a Zahn-novel cover or Rebels-animation portrait. The point is to differentiate from canon.

---

## After downloading

```bash
# Confirm count
cd data/images
ls *.jpg | wc -l       # should print 70

# Find anything that doesn't match the expected ID pattern
ls | grep -vE '^(c_|s_|p_|e_).*\.(jpg|png|jpeg|webp)$'
# Should print nothing.

# Find missing IDs (compare against CSV)
cd ../..
python -c "
import pandas as pd
from pathlib import Path
df = pd.read_csv('data/entities.csv')
files = {f.stem for f in Path('data/images').glob('*')}
missing = [eid for eid in df['entity_id'] if eid not in files]
print('Missing images for:', missing if missing else 'none — all good')
"
```

If both checks pass, run `python -m src.ingest`.
