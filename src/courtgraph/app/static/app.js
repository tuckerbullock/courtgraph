'use strict';
const $ = id => document.getElementById(id);
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num = value => Number(value).toLocaleString(undefined, {maximumFractionDigits: 0});
const dec = value => Number(value).toFixed(1);
const signed = value => `${value >= 0 ? '+' : ''}${dec(value)}`;
let state, lineups = [], selectedId = '', observationRequest = 0, revision = 0;
async function api(path, payload) {
  const response = await fetch(path, payload === undefined ? {} : {method: 'POST', headers: {'Content-Type': 'application/json', 'X-CourtGraph-Request': 'local'}, body: JSON.stringify(payload)});
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || 'Could not load this result.');
  return result;
}
function showError(id, error) { $(id).textContent = error.message || String(error); $(id).hidden = false; }
function mode(name) {
  $('real-view').hidden = name !== 'real'; $('synthetic-view').hidden = name !== 'synthetic';
  document.body.classList.toggle('is-synthetic', name === 'synthetic');
  for (const key of ['real', 'synthetic']) {
    $(`nav-${key}`).classList.toggle('active', key === name);
    if (key === name) $(`nav-${key}`).setAttribute('aria-current', 'page'); else $(`nav-${key}`).removeAttribute('aria-current');
  }
  $('breadcrumb').textContent = name === 'real' ? 'Game explorer' : 'Lineup sandbox';
}
function options(id, items, label) {
  $(id).replaceChildren(new Option(label, ''), ...items.map(item => new Option(item.name, item.id)));
}
function metric(label, value, note) { return `<div class="metric"><span class="metric-label">${esc(label)}</span><span class="metric-value">${num(value)}</span><span class="metric-sub">${esc(note)}</span></div>`; }
function gameMatchup(game) {
  if (game.away_team && game.home_team) return `${game.away_team} vs ${game.home_team}`;
  const teams = game.scores.map(score => score.team);
  return teams.length ? teams.join(' vs ') : `Game ${game.id}`;
}
function coverageSummary(real) {
  const c = real.coverage;
  $('coverage-summary').innerHTML = `<div><span class="eyebrow">LOCAL ARCHIVE COVERAGE</span><strong>${num(c.archive_games)} games found</strong><p>${num(c.complete_games)} had all required source files; ${num(c.attempted_games)} were attempted.</p></div><div class="coverage-flow"><span><b>${num(c.accepted_games)}</b> accepted</span><span><b>${num(c.quarantined_games)}</b> quarantined</span><span><b>${num(c.source_incomplete_games)}</b> incomplete source</span></div>`;
}
async function loadObservations() {
  const request = ++observationRequest;
  const query = new URLSearchParams({game: $('game-filter').value, team: $('team-filter').value, player: $('player-filter').value, minimum: $('minimum-filter').value});
  $('global-error').hidden = true;
  try {
    const result = await api(`/api/observations?${query}`);
    if (request !== observationRequest) return;
    lineups = result.lineups;
    $('real-metrics').innerHTML = metric('Games in this selection', result.games, 'With accepted stints') + metric('Accepted possessions', result.possessions, 'Across selected offensive records') + metric('Observed lineups', lineups.length, 'Meeting your minimum sample') + metric('Stint records', result.stints, 'Before minimum-sample filtering');
    $('lineup-count').textContent = `${num(lineups.length)} combinations · select a lineup to inspect its evidence`;
    selectedId = lineups.some(row => row.id === selectedId) ? selectedId : (lineups[0]?.id || '');
    renderLineups();
  } catch (error) { if (request === observationRequest) { lineups = []; selectedId = ''; renderLineups(); $('real-metrics').replaceChildren(); showError('global-error', error); } }
}
function renderLineups() {
  const field = $('sort-lineups').value;
  const rows = [...lineups].sort((a,b) => b[field] - a[field] || a.id.localeCompare(b.id));
  $('lineup-rows').innerHTML = rows.map(row => `<tr class="${row.id === selectedId ? 'selected-row' : ''}"><td><span class="lineup-team">${esc(row.team)} · ${row.games} GAME${row.games === 1 ? '' : 'S'}</span><span class="lineup-names">${row.players.map(esc).join(' · ')}</span></td><td class="number">${num(row.possessions)}</td><td class="number rate">${dec(row.rating)}</td><td><button type="button" class="inspect" data-lineup="${esc(row.id)}" aria-label="Inspect ${esc(row.players.join(', '))}">Inspect</button></td></tr>`).join('') || '<tr><td colspan="4"><p>No lineups match these filters. Try a lower minimum or reset the filters. Quarantined games have no usable lineups.</p></td></tr>';
  const row = lineups.find(x => x.id === selectedId);
  if (!row) { $('lineup-detail').innerHTML = '<h2>No lineup selected</h2><p>Choose a game with accepted records or widen your filters.</p>'; return; }
  $('lineup-detail').innerHTML = `<h2>${esc(row.team)}</h2><p>${row.players.map(esc).join('<br>')}</p><div class="rate-big">${dec(row.rating)}</div><p>Observed offensive points / 100</p><div class="detail-line"><span>Sample</span><b>${num(row.possessions)} possessions</b></div><div class="detail-line"><span>Scoring</span><b>${num(row.points)} points</b></div><div class="detail-line"><span>Coverage</span><b>${row.games} games / ${row.stints} stints</b></div><div class="detail-line"><span>Opponents</span><b>${row.opponents.map(esc).join(', ')}</b></div><div class="detail-line"><span>Dates</span><b>${row.dates.map(esc).join('<br>')}</b></div><div class="detail-line"><span>Garbage-time flags</span><b>${row.downweighted_stints} stints</b></div><p class="sample-warning">Small, selected sample. This rate does not estimate chemistry, player talent, or future performance. Flagged garbage-time stints remain fully counted here.</p>`;
}
function renderQuality() {
  const real = state.real;
  coverageSummary(real);
  const failures = real.games.filter(game => game.status !== 'accepted');
  $('quality-games').innerHTML = failures.map(game => `<article class="game-card"><div class="game-date">${esc(game.date || 'Date unavailable')} <span class="game-status quarantined">${esc(game.status.replace('_', ' ').toUpperCase())}</span></div><h3>${esc(gameMatchup(game))}</h3>${game.scores.filter(score => score.points !== null).map(score => `<div class="score-line"><span>${esc(score.team)}</span><b>${num(score.points)}</b></div>`).join('')}<p>${num(game.possessions)} accepted possessions · ${num(game.stints)} stints</p><p>${game.score_matched === false ? 'Score mismatch flagged by the importer.' : 'No accepted reconstruction is available.'}</p>${game.quarantine_reason ? `<p>Reason: ${esc(game.quarantine_reason)}</p>` : ''}<details><summary>Source, flags & exclusions</summary><p>${esc(game.score_source)}</p><ul>${Object.entries(game.exclusions).map(([reason,count]) => `<li>${esc(reason)}: ${num(count)}</li>`).join('') || '<li>No possession-level exclusions recorded.</li>'}${game.flags.map(flag => `<li>Flag: ${esc(flag)}</li>`).join('')}</ul></details></article>`).join('') || '<p class="all-accepted">Every attempted game was accepted.</p>';
  const c = real.coverage;
  const entries = [['Recorded source', real.source], ['Archive coverage', `${c.archive_games} games found; ${c.complete_games} complete inputs; ${c.attempted_games} attempted; ${c.accepted_games} accepted`], ['Data through', real.cutoff || 'No accepted dates'], ['Converter', real.converter], ['Ingest generated', real.created_utc], ['Parser', `${real.parser.tool || 'Unknown'} ${real.parser.version || ''}`], ['Source commit', real.source_commit], ['Verified stint SHA-256', real.checksum], ['Use', 'Local demonstration only. No redistribution. Participants are not complete dated rosters.']];
  $('provenance').innerHTML = entries.map(([key,value]) => `<dt>${esc(key)}</dt><dd>${esc(value)}</dd>`).join('');
}
function fillSlots(side, label, pool, offsetPick) {
  const container = $(`predict-${side}-selects`);
  container.innerHTML = Array.from({length:5}, (_,i) => `<div class="player-slot"><span aria-hidden="true">${String(i+1).padStart(2,'0')}</span><label><span class="slot-label">${label} player ${i+1}</span><select id="predict-${side}-${i}" required></select></label></div>`).join('');
  for (let i=0;i<5;i++) { $(`predict-${side}-${i}`).replaceChildren(...pool.players.map(p => new Option(`${p.name} (${num(p.possessions)} poss.)`, p.id))); $(`predict-${side}-${i}`).selectedIndex = Math.min(i + offsetPick, pool.players.length - 1); }
}
async function loadTeamPlayers(side) {
  const team = $(`predict-${side}-team`).value;
  $('predict-error').hidden = true;
  if (!team) { $(`predict-${side}-selects`).innerHTML = ''; if (side === 'off') $('predict-altb-selects').innerHTML = ''; return; }
  try {
    const pool = await api(`/api/player-pool?team=${encodeURIComponent(team)}`);
    if (side === 'off') {
      fillSlots('off', 'Lineup A', pool, 0);
      // Lineup B defaults to a visibly different five (shifted by one)
      fillSlots('altb', 'Lineup B', pool, 1);
    } else {
      fillSlots('def', 'Defense', pool, 0);
    }
  } catch (error) { $(`predict-${side}-selects`).innerHTML = ''; if (side === 'off') $('predict-altb-selects').innerHTML = ''; showError('predict-error', error); }
}
function predictSelected(side) { return Array.from({length:5}, (_,i) => Number($(`predict-${side}-${i}`)?.value)); }
function predictContext() { return {home: $('predict-home').checked, playoff: $('predict-playoff').checked, rest: Number($('predict-rest').value)}; }
function unseenWarning(support) {
  const u = [...support.unseen_offense_players, ...support.unseen_defense_players];
  return u.length ? `<p class="sample-warning">Never observed in the training data — additive-only, no talent estimate: ${u.join(', ')}</p>` : '';
}
async function runPredict(fn, button, otherButton, work) {
  $('predict-error').hidden = true;
  button.disabled = true; otherButton.disabled = true;
  const label = button.textContent;
  button.textContent = 'Fitting rung 3 on first use — this can take a minute…';
  try { await work(); }
  catch (error) { $('predict-results').replaceChildren(); showError('predict-error', error); }
  finally { button.disabled = false; otherButton.disabled = false; button.textContent = label; }
}
async function predictLineup() {
  await runPredict(predictLineup, $('predict-button'), $('predict-compare-button'), async () => {
    const offense = predictSelected('off'), defense = predictSelected('def');
    if (offense.some(Number.isNaN) || defense.some(Number.isNaN)) throw new Error('Choose an offense team and a defense team first.');
    const r = await api('/api/predict-real', {offense, defense, ...predictContext()});
    $('predict-results').innerHTML = `<div class="comparison-top"><div><span class="eyebrow">MODEL ESTIMATE / REAL DATA</span><h2>Additive prediction.</h2><p>Offensive points per 100 possessions. Not net rating.</p></div></div><div class="table-scroll"><table><thead><tr><th>Component</th><th>Value</th></tr></thead><tbody><tr><td>Talent (individual + baseline)</td><td class="number">${dec(r.talent)}</td></tr><tr><td>Context</td><td class="number">${dec(r.context_value)}</td></tr><tr class="total"><td>Total predicted value</td><td class="number">${dec(r.total)}</td></tr></tbody></table></div><div class="uncertainty"><strong>Calibrated predictive intervals:</strong> 80% [${signed(r.interval_80[0])}, ${signed(r.interval_80[1])}] &nbsp;·&nbsp; 95% [${signed(r.interval_95[0])}, ${signed(r.interval_95[1])}].<br>${esc(r.note)}</div>${unseenWarning(r.support)}<p class="footnote">Weakest offensive exposure in this lineup: ${num(r.support.min_offense_player_possessions)} possessions.</p>`;
  });
}
async function compareLineups() {
  await runPredict(compareLineups, $('predict-compare-button'), $('predict-button'), async () => {
    const offense = predictSelected('off'), alternative = predictSelected('altb'), defense = predictSelected('def');
    if (offense.some(Number.isNaN) || alternative.some(Number.isNaN) || defense.some(Number.isNaN)) throw new Error('Fill Lineup A, Lineup B, and the opponent.');
    const r = await api('/api/compare-real', {offense, alternative, defense, ...predictContext()});
    const a = r.a, b = r.b, d = r.delta;
    const better = key => Math.abs(d[key]) < 0.05 ? 'about equal' : (d[key] > 0 ? 'Lineup B' : 'Lineup A');
    $('predict-results').innerHTML = `<div class="comparison-top"><div><span class="eyebrow">MODEL ESTIMATE / REAL DATA</span><h2>The difference in your five.</h2><p>Offensive points per 100 possessions, same opponent and context. Not net rating.</p></div><div class="delta-pill">B − A &nbsp; <strong>${signed(d.total)}</strong> / 100</div></div><div class="table-scroll"><table><thead><tr><th>Component</th><th>Lineup A</th><th>Lineup B</th><th>B − A</th></tr></thead><tbody><tr><td>Talent</td><td class="number">${dec(a.talent)}</td><td class="number">${dec(b.talent)}</td><td class="number">${signed(d.talent)}</td></tr><tr><td>Context</td><td class="number">${dec(a.context_value)}</td><td class="number">${dec(b.context_value)}</td><td class="number">${signed(d.context_value)}</td></tr><tr class="total"><td>Total</td><td class="number">${dec(a.total)}</td><td class="number">${dec(b.total)}</td><td class="number">${signed(d.total)}</td></tr></tbody></table></div><div class="result-notes"><div><h3>Higher predicted offensive value</h3><p><strong>${better('total')}</strong> — neither is a validated recommendation.</p></div><div><h3>Per-lineup 95% intervals</h3><p>A [${signed(a.interval_95[0])}, ${signed(a.interval_95[1])}]<br>B [${signed(b.interval_95[0])}, ${signed(b.interval_95[1])}]</p></div></div><div class="uncertainty">${esc(r.delta_note)}<br>${esc(r.note)}</div>${unseenWarning(a.support)}${unseenWarning(b.support)}`;
  });
}
function buildSelectors() {
  for (const [key,title] of [['offense','Lineup A'],['alternative','Lineup B'],['defense','Opponent']]) {
    $(`${key}-selects`).innerHTML = Array.from({length:5}, (_,i) => `<div class="player-slot"><span aria-hidden="true">${String(i+1).padStart(2,'0')}</span><label><span class="slot-label">${title} player ${i+1}</span><select id="${key}-${i}" required></select></label></div>`).join('');
    for (let i=0; i<5; i++) $(`${key}-${i}`).replaceChildren(...state.synthetic.players.map(p => new Option(p.name,p.id)));
  }
  resetLineups();
  const syn = state.synthetic;
  $('model-evidence').innerHTML = `<p><strong>${esc(syn.model)}</strong><br>${num(syn.training_stints)} synthetic stints · ${num(syn.training_possessions)} training possessions · seed ${syn.seed} · synthetic data through ${esc(syn.cutoff)}.</p><p>The talent term includes the baseline scoring level, offensive player effects, and opposing defensive player effects. The interaction term is centered against the model's reference lineup sample; it is model-dependent, not interpersonal chemistry.</p><p>${syn.bootstrap_members} game-block bootstrap members provide an approximate 80% interaction interval. This does not include every source of uncertainty, is not a calibrated total-performance interval, and is not evidence that either lineup is better in the NBA.</p><p>Seen means the exact offensive five appeared in synthetic training. Partially seen means its pairs were seen but the five were not. Unseen can mean some teammate pairs were never observed together. Exposure counts refer to individual offensive training possessions, not possessions played by this exact five.</p><p>Only the synthetic model is trained when the app starts; real game records never enter it. The app does not run or claim a new held-out validation study.</p>`;
}
function selected(key) { return Array.from({length:5}, (_,i) => Number($(`${key}-${i}`).value)); }
function setLineup(key, ids) { ids.forEach((id,i) => { $(`${key}-${i}`).value = String(id); }); }
function updateChoices() {
  const offense = selected('offense'), alternative = selected('alternative'), defense = selected('defense');
  for (const key of ['offense','alternative','defense']) {
    const own = selected(key), forbidden = new Set(key === 'defense' ? [...offense,...alternative] : defense);
    for (let i=0;i<5;i++) {
      const select = $(`${key}-${i}`);
      for (const option of select.options) option.disabled = (own.includes(Number(option.value)) && Number(option.value) !== own[i]) || forbidden.has(Number(option.value));
    }
  }
}
function dirty() { revision++; updateChoices(); $('comparison-results').innerHTML = '<h2>Ready for a new comparison</h2><p>Your inputs changed. Compare again to update the estimates.</p>'; $('compare-error').hidden = true; }
function resetLineups() {
  for (const key of ['offense','alternative','defense']) setLineup(key,state.synthetic[key]);
  $('home-context').checked = true; $('playoff-context').checked = false; $('rest-context').value = '1'; dirty();
}
async function compare(event) {
  event?.preventDefault();
  if (!$('comparison-form').reportValidity()) return;
  const version = revision;
  $('compare-button').disabled = true; $('compare-button').textContent = 'Comparing…'; $('compare-error').hidden = true;
  try {
    const data = await api('/api/compare', {offense:selected('offense'), alternative:selected('alternative'), defense:selected('defense'), home:$('home-context').checked, playoff:$('playoff-context').checked, rest:Number($('rest-context').value)});
    if (version !== revision) return;
    const [a,b] = data.results, delta = data.delta;
    const rows = [['talent','Individual talent + scoring baseline'],['interaction','Interaction surplus'],['context','Context'],['total','Predicted offensive points / 100']];
    const choice = key => Math.abs(delta[key]) < 0.05 ? 'Approximately equal' : `Lineup ${delta[key] > 0 ? 'B' : 'A'}`;
    $('comparison-results').innerHTML = `<div class="comparison-top"><div><span class="eyebrow">MODEL ESTIMATES / SYNTHETIC</span><h2>The difference in your five.</h2><p>Offensive points per 100 possessions. This is not net rating or a game forecast.</p></div><div class="delta-pill">B − A &nbsp; <strong>${signed(delta.total)}</strong> / 100</div></div><div class="table-scroll"><table><thead><tr><th>Component</th><th>Lineup A</th><th>Lineup B</th><th>B − A</th></tr></thead><tbody>${rows.map(([key,label]) => `<tr class="${key === 'total' ? 'total' : ''}"><td>${label}</td><td>${dec(a.decomposition[key])}</td><td>${dec(b.decomposition[key])}</td><td>${signed(delta[key])}</td></tr>`).join('')}</tbody></table></div><div class="result-notes"><div><h3>Separate the questions</h3><p>Higher predicted offensive value: <strong>${choice('total')}</strong></p><p>Higher predicted interaction surplus: <strong>${choice('interaction')}</strong></p><p>These can favor different lineups. Neither is a validated recommendation.</p></div><div><h3>Training support</h3>${[a,b].map((r,i) => `<p><strong>Lineup ${i ? 'B':'A'}</strong>: ${esc(r.decomposition.offense_novelty)} · weakest individual exposure ${num(r.support.min_player_possessions)} offensive possessions</p>`).join('')}</div></div><div class="uncertainty"><strong>80% approximate interaction intervals:</strong> A [${signed(a.interval.lower)}, ${signed(a.interval.upper)}] · B [${signed(b.interval.lower)}, ${signed(b.interval.upper)}].<br>${esc(data.uncertainty_note)}</div>`;
  } catch(error) { if (version === revision) { $('comparison-results').replaceChildren(); showError('compare-error',error); } }
  finally { $('compare-button').disabled = false; $('compare-button').textContent = 'Compare lineups →'; }
}
$('nav-real').addEventListener('click', () => mode('real'));
$('nav-synthetic').addEventListener('click', () => mode('synthetic'));
$('empty-sandbox').addEventListener('click', () => mode('synthetic'));
for (const id of ['game-filter','team-filter','player-filter','minimum-filter']) $(id).addEventListener('change', loadObservations);
$('reset-filters').addEventListener('click', () => { for (const id of ['game-filter','team-filter','player-filter']) $(id).value = ''; $('minimum-filter').value = '1'; loadObservations(); });
$('sort-lineups').addEventListener('change', renderLineups);
$('lineup-rows').addEventListener('click', event => { const button = event.target.closest('button[data-lineup]'); if (button) { selectedId = button.dataset.lineup; renderLineups(); } });
$('comparison-form').addEventListener('submit', compare);
$('comparison-form').addEventListener('change', dirty);
$('copy-a').addEventListener('click', () => { setLineup('alternative',selected('offense')); dirty(); });
$('reset-lineups').addEventListener('click', resetLineups);
$('predict-off-team').addEventListener('change', () => loadTeamPlayers('off'));
$('predict-def-team').addEventListener('change', () => loadTeamPlayers('def'));
$('predict-button').addEventListener('click', predictLineup);
$('predict-compare-button').addEventListener('click', compareLineups);
async function start() {
  try {
    state = await api('/api/state'); buildSelectors();
    $('real-empty').hidden = state.real.loaded; $('real-content').hidden = !state.real.loaded;
    if (state.real.loaded) {
      options('game-filter',state.real.games.map(g => ({id:g.id,name:`${g.date || 'date unavailable'} · ${gameMatchup(g)}${g.status === 'accepted' ? '' : ` · ${g.status.replace('_', ' ')}`}`})),`All ${state.real.coverage.archive_games} archive games`);
      options('team-filter',state.real.teams,'All teams'); options('player-filter',[...state.real.players].sort((a,b)=>a.name.localeCompare(b.name)),'Any player');
      options('predict-off-team',state.real.teams,'Choose a team'); options('predict-def-team',state.real.teams,'Choose a team');
      renderQuality(); await loadObservations();
    }
    $('loading').hidden = true; mode('real'); await compare();
  } catch(error) { $('loading').hidden = true; showError('global-error',error); }
}
start();
