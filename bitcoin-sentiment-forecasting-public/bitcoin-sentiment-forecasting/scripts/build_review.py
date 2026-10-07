"""Build a private offline review page; provider text stays under work/."""
import csv
import html
import json
from collections import defaultdict
from pathlib import Path

root = Path(__file__).resolve().parents[1]
folder = root / "work/private_news"
raw = {r["article_id"]: r for r in csv.DictReader((folder / "news_articles.csv").open(encoding="utf-8"))}
scores = list(csv.DictReader((folder / "processed/bitcoin_sentiment.csv").open(encoding="utf-8")))
unscored = json.loads((folder / "processed/unscored.json").read_text())
selected = {}
for label in ("positive", "negative", "neutral"):
    for row in sorted([r for r in scores if r["predicted_label"] == label], key=lambda r: abs(float(r["compound_score"])), reverse=True)[:3]:
        selected[row["article_id"]] = (row, "Strong " + label + " example" if label != "neutral" else "Neutral example")
for row in unscored[:3]:
    selected[row["article_id"]] = (row, "Unresolved Bitcoin relevance")
groups = defaultdict(list)
for row in scores:
    groups[row["duplicate_group_id"]].append(row)
for group in groups.values():
    if len(group) > 1:
        for row in group:
            selected[row["article_id"]] = (row, "Possible duplicate: " + row["duplicate_group_id"])

notes = {
    "news_376eb9d9c33975791561": "Review focus: Bitcoin rises, but weak macroeconomic language may drive a negative lexical score. Choose a label for Bitcoin-specific language, not future returns.",
    "news_4cc76704251914550863": "Review focus: promotional framing and a multi-asset passage may inflate positivity or mix Ethereum language with Bitcoin evidence.",
    "news_921c8782779a017e3406": "Review focus: the passage describes rising prices but receives a neutral lexical score. Decide whether your annotation policy treats factual price rises as sentiment.",
}
cards = []
for identity, (row, reason) in selected.items():
    article = raw[identity]
    escape = html.escape
    url = article["source_url"]
    if not url.startswith(("https://", "http://")):
        url = "#"
    evidence = row.get("evidence_text", article["summary"])
    cards.append(f'''<article data-id="{escape(identity)}"><p class="badge">{escape(reason)}</p>
<h2>{escape(article['headline'])}</h2><p><a href="{escape(url, quote=True)}" target="_blank" rel="noopener noreferrer">Original source</a> · {escape(article['source_name'])}</p>
<p>Model: <b>{escape(row.get('predicted_label', 'not scored'))}</b> · score: {escape(row.get('compound_score', '—'))}</p>
<blockquote>{escape(evidence)}</blockquote><p class="note">{escape(notes.get(identity, 'Check Bitcoin relevance, quoted versus author language, and whether the extracted evidence supports this score.'))}</p>
<label>Bitcoin relevance <select class="relevance"><option value="pending">Not reviewed</option><option>relevant</option><option>unclear</option><option>irrelevant</option></select></label>
<label>Your sentiment label <select class="label"><option value="pending">Not reviewed</option><option>positive</option><option>negative</option><option>neutral</option><option>mixed</option><option>insufficient_evidence</option></select></label>
<label>Duplicate review <select class="duplicate"><option value="pending">Not reviewed</option><option>same_story</option><option>different_story</option><option>not_applicable</option></select></label>
<label>Notes <textarea placeholder="Explain your judgment; avoid pasting additional article text."></textarea></label></article>''')
page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Bitcoin news review</title><style>body{font:17px system-ui;max-width:950px;margin:40px auto;padding:0 20px;background:#f4f6f8;color:#192635}article{background:white;padding:24px;margin:24px 0;border-radius:12px;border:1px solid #dbe2e8}h2{font-size:21px}blockquote{margin:15px 0;padding:16px;background:#eef3f7;border-left:4px solid #497a99}label{display:block;margin:12px 0}select,textarea,button{font:inherit;padding:8px}textarea{display:block;width:95%;min-height:65px}.note{color:#714816}.badge{font-size:14px;color:#476378}button{background:#194f71;color:white;border:0;border-radius:6px;cursor:pointer}</style>
<h1>Review Bitcoin news sentiment</h1><p>Private, local development review. This is a deliberately selected sample, not an unbiased sentiment-accuracy evaluation.</p>
<p>Judge the language about Bitcoin, not whether its price will rise. Distinguish promotional or quoted opinions from factual market movement. Uncertain cases can remain mixed or insufficient evidence.</p>
<p>Changes stay on this page until you download them. Keep the downloaded review private. Refreshing discards unsaved selections.</p>
<p id="progress">0 examples reviewed</p><button onclick="save()">Download my review</button>'''+"".join(cards)+'''
<button onclick="save()">Download my review</button><script>
function reviewed(a){return a.querySelector('.relevance').value!=='pending' && a.querySelector('.label').value!=='pending';}
function progress(){const cards=Array.from(document.querySelectorAll('article'));document.getElementById('progress').textContent=cards.filter(reviewed).length+' of '+cards.length+' examples reviewed';}
document.addEventListener('change',progress);
function save(){const cards=Array.from(document.querySelectorAll('article'));if(!cards.some(reviewed)){alert('Choose Bitcoin relevance and a sentiment label for at least one example before downloading.');return;}const records=cards.map(a=>({article_id:a.dataset.id,relevance:a.querySelector('.relevance').value,human_sentiment:a.querySelector('.label').value,duplicate_judgment:a.querySelector('.duplicate').value,notes:a.querySelector('textarea').value}));const payload={reviewed_at:new Date().toISOString(),purpose:'private targeted development review; not held-out accuracy evaluation',records};const u=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:'application/json'}));const link=document.createElement('a');link.href=u;link.download='bitcoin-news-human-review.json';link.click();setTimeout(()=>URL.revokeObjectURL(u),1000);}
</script></html>'''
target = folder / "news-review.html"
target.write_text(page, encoding="utf-8")
print(f"Private review page created with {len(selected)} examples.")
