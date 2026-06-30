/**
 * app.js — CupCast26
 * Bracket oficial FIFA 2026 · Modal H2H · Resultado real en cards
 */
'use strict';

// ── Cuadro oficial FIFA 2026 ──────────────────────────────────
// Fuente: bracketmundial2026.com / FIFA
// Mitad izquierda: M73,M74,M75,M76,M77,M78,M79,M80
// Mitad derecha:   M81,M82,M83,M84,M85,M86,M87,M88
// Octavos: M89=w74vw77, M90=w73vw75, M91=w76vw78, M92=w79vw80
//          M93=w83vw84, M94=w81vw82, M95=w86vw88, M96=w85vw87
// Cuartos: M97=w89vw90, M98=w93vw94, M99=w91vw92, M100=w95vw96
// Semis:   M101=w97vw98, M102=w99vw100  Final: M104=w101vw102

// Índice de bracket por número de partido → {team_a, team_b}
const BRACKET_DEF = {
  // 16avos — lado izquierdo (ramas que van hacia la derecha hacia el trofeo)
  M73:{ a:'South Africa',         b:'Canada'                   },
  M74:{ a:'Germany',              b:'Paraguay'                 },
  M75:{ a:'Netherlands',          b:'Morocco'                  },
  M76:{ a:'Brazil',               b:'Japan'                    },
  M77:{ a:'France',               b:'Sweden'                   },
  M78:{ a:'Ivory Coast',          b:'Norway'                   },
  M79:{ a:'Mexico',               b:'Ecuador'                  },
  M80:{ a:'England',              b:'DR Congo'                 },
  // 16avos — lado derecho
  M81:{ a:'United States',        b:'Bosnia and Herzegovina'   },
  M82:{ a:'Belgium',              b:'Senegal'                  },
  M83:{ a:'Portugal',             b:'Croatia'                  },
  M84:{ a:'Spain',                b:'Austria'                  },
  M85:{ a:'Switzerland',          b:'Algeria'                  },
  M86:{ a:'Argentina',            b:'Cape Verde'               },
  M87:{ a:'Colombia',             b:'Ghana'                    },
  M88:{ a:'Australia',            b:'Egypt'                    },
};

// Estructura del cuadro: qué partidos se enfrentan en cada ronda
// Cada par = [partido_superior, partido_inferior] → ganadores se cruzan
const BRACKET_TREE = {
  left: [
    // Octavos izq (M89=w74vw77, M90=w73vw75, M91=w76vw78, M92=w79vw80)
    { r16: ['M74','M77'], r8: 'M89' },
    { r16: ['M73','M75'], r8: 'M90' },
    { r16: ['M76','M78'], r8: 'M91' },
    { r16: ['M79','M80'], r8: 'M92' },
  ],
  right: [
    // Octavos der (M93=w83vw84, M94=w81vw82, M95=w86vw88, M96=w85vw87)
    { r16: ['M83','M84'], r8: 'M93' },
    { r16: ['M81','M82'], r8: 'M94' },
    { r16: ['M86','M88'], r8: 'M95' },
    { r16: ['M85','M87'], r8: 'M96' },
  ],
  // Cuartos: M97=w89vw90, M98=w93vw94, M99=w91vw92, M100=w95vw96
  qf_left:  [{ r8s:['M89','M90'], qf:'M97' }, { r8s:['M91','M92'], qf:'M99' }],
  qf_right: [{ r8s:['M93','M94'], qf:'M98' }, { r8s:['M95','M96'], qf:'M100'}],
  // Semis: M101=w97vw98, M102=w99vw100  Final: M104=w101vw102
  sf_left: 'M101', sf_right: 'M102',
};

const ROUND_KEYS = ['Round of 32','Round of 16','Quarter-finals','Semi-finals','Final','Winner'];
const ROUND_ES   = {'Round of 32':'16avos','Round of 16':'8avos','Quarter-finals':'Cuartos','Semi-finals':'Semis','Final':'Final','Winner':'Campeón'};

const API = { status:'/api/status', predictions:'/api/predictions', montecarlo:'/api/montecarlo', elo:'/api/elo', bracket:'/api/bracket', accuracy:'/api/accuracy' };

let eloChart = null;
let globalProcessed = null;  // cache de predictions para modal H2H

document.addEventListener('DOMContentLoaded', () => { loadAll(); setInterval(loadAll, 10*60*1000); });

async function loadAll() {
  try {
    const [status, predictions, mc, elo, bracket, accuracy] = await Promise.all([
      fetchJSON(API.status), fetchJSON(API.predictions),
      fetchJSON(API.montecarlo), fetchJSON(API.elo),
      fetchJSON(API.bracket),   fetchJSON(API.accuracy),
    ]);
    globalProcessed = predictions;
    renderStatus(status);
    renderBracket(predictions, mc, accuracy);
    renderMatches(predictions, accuracy);
    renderHeatmap(mc);
    renderElo(elo);
    renderAccuracy(accuracy);
  } catch(e) {
    console.error(e);
    document.getElementById('statusDot').className = 'pulse err';
    document.getElementById('statusText').textContent = 'Sin conexión';
  }
}

async function fetchJSON(url) {
  const r = await fetch(url); if (!r.ok) throw new Error(url+' → '+r.status); return r.json();
}

// ══════════════════════════════════════════════════
// STATUS
// ══════════════════════════════════════════════════
function renderStatus(s) {
  const dot = document.getElementById('statusDot');
  const txt = document.getElementById('statusText');
  const upd = document.getElementById('lastUpdate');
  const rnd = document.getElementById('heroRound');
  if (s.status==='ready'){dot.className='pulse live';txt.textContent='En vivo';}
  else{dot.className='pulse';txt.textContent='Actualizando…';}
  if (s.last_update){ const d=new Date(s.last_update+'Z'); upd.textContent='· '+d.toLocaleTimeString('es-ES',{hour:'2-digit',minute:'2-digit'}); }
  if (rnd&&s.current_round) rnd.textContent=s.current_round;
}

// ══════════════════════════════════════════════════
// HELPERS: probabilidades y ganadores
// ══════════════════════════════════════════════════
function buildPredIdx(predictions) {
  const idx = {};
  (predictions.matches||[]).forEach(m => {
    idx[m.team_a+'|'+m.team_b] = m;
    idx[m.team_b+'|'+m.team_a] = {...m, team_a:m.team_b, team_b:m.team_a, prob_a:m.prob_b, prob_b:m.prob_a,
      favorite:m.team_b===m.favorite?m.team_b:m.team_a, p_elo_a:1-m.p_elo_a, p_fifa_a:1-m.p_fifa_a, p_h2h_a:1-m.p_h2h_a };
  });
  return idx;
}

function buildMcIdx(mc) {
  const idx = {};
  (mc.teams||[]).forEach(t => { idx[t.team]=t; });
  return idx;
}

function buildAccIdx(accuracy) {
  const idx = {};
  (accuracy.match_results||[]).forEach(r => {
    idx[r.team_a+'|'+r.team_b] = r;
    idx[r.team_b+'|'+r.team_a] = r;
  });
  return idx;
}

function getProb(predIdx, a, b) {
  const p = predIdx[a+'|'+b]; return p ? p.prob_a : 0.5;
}

function projectedWinner(predIdx, a, b) {
  return getProb(predIdx,a,b) >= 0.5 ? a : b;
}

function getRoundProb(mcIdx, team, roundKey) {
  const t = mcIdx[team]; if (!t) return null;
  return t[roundKey] !== undefined ? t[roundKey] : null;
}

// ══════════════════════════════════════════════════
// BRACKET — 8 izquierda + 8 derecha
// ══════════════════════════════════════════════════
function renderBracket(predictions, mc, accuracy) {
  const container = document.getElementById('bracketOuter');
  container.innerHTML = '';

  const predIdx = buildPredIdx(predictions);
  const mcIdx   = buildMcIdx(mc);
  const accIdx  = buildAccIdx(accuracy);

  // Calcular ganadores proyectados de cada partido
  const winner = {};
  Object.entries(BRACKET_DEF).forEach(([id, m]) => {
    // Si hay resultado real registrado en accuracy, usarlo
    const real = accIdx[m.a+'|'+m.b];
    winner[id] = real ? real.real_winner : projectedWinner(predIdx, m.a, m.b);
  });
  // Octavos
  const octavos = {
    M89: projectedWinner(predIdx, winner['M74'], winner['M77']),
    M90: projectedWinner(predIdx, winner['M73'], winner['M75']),
    M91: projectedWinner(predIdx, winner['M76'], winner['M78']),
    M92: projectedWinner(predIdx, winner['M79'], winner['M80']),
    M93: projectedWinner(predIdx, winner['M83'], winner['M84']),
    M94: projectedWinner(predIdx, winner['M81'], winner['M82']),
    M95: projectedWinner(predIdx, winner['M86'], winner['M88']),
    M96: projectedWinner(predIdx, winner['M85'], winner['M87']),
  };
  // Cuartos
  const cuartos = {
    M97:  projectedWinner(predIdx, octavos.M89, octavos.M90),
    M98:  projectedWinner(predIdx, octavos.M93, octavos.M94),
    M99:  projectedWinner(predIdx, octavos.M91, octavos.M92),
    M100: projectedWinner(predIdx, octavos.M95, octavos.M96),
  };
  // Semis
  const sfL = projectedWinner(predIdx, cuartos.M97, cuartos.M98);
  const sfR = projectedWinner(predIdx, cuartos.M99, cuartos.M100);
  // Final
  const champion = projectedWinner(predIdx, sfL, sfR);

  // ── Mitad izquierda ──────────────────────────────
  const leftHalf = document.createElement('div');
  leftHalf.className = 'bracket-half bracket-half--left';

  // Columna 16avos izq
  leftHalf.appendChild(buildR16Col(
    ['M74','M73','M76','M79','M77','M78','M80'], // no — respetamos pares
    [['M74','M73'],['M76','M77'],['M78','M79'],['M80','M80']], // placeholder
    predIdx, mcIdx, accIdx, winner, 'left'
  ));

  // ── construir columnas izq de forma correcta
  leftHalf.innerHTML = '';
  leftHalf.appendChild(buildHalfCols(
    [['M73','M74'],['M75','M76'],['M77','M78'],['M79','M80']],
    [['M89','M90'],['M91','M92']],
    [['M97','M99']],
    [sfL],
    predIdx, mcIdx, accIdx, winner, octavos, cuartos, 'left'
  ));

  // ── Trofeo central ───────────────────────────────
  const center = document.createElement('div');
  center.className = 'bracket-center';
  center.innerHTML = `
    <div class="b-trophy__icon">🏆</div>
    <div class="b-final-teams">
      ${buildFinalTeamEl(sfL, predIdx, mcIdx, 'Final', true)}
      ${buildFinalTeamEl(sfR, predIdx, mcIdx, 'Final', sfL===champion)}
    </div>
    <div class="b-trophy__label">Final<br/>19 Jul · MetLife</div>`;

  // ── Mitad derecha ────────────────────────────────
  const rightHalf = document.createElement('div');
  rightHalf.className = 'bracket-half bracket-half--right';
  rightHalf.appendChild(buildHalfCols(
    [['M81','M82'],['M83','M84'],['M85','M86'],['M87','M88']],
    [['M93','M94'],['M95','M96']],
    [['M98','M100']],
    [sfR],
    predIdx, mcIdx, accIdx, winner, octavos, cuartos, 'right'
  ));

  container.appendChild(leftHalf);
  container.appendChild(center);
  container.appendChild(rightHalf);
}

function buildHalfCols(r16Pairs, r8Pairs, qfPairs, sfTeams, predIdx, mcIdx, accIdx, winner, octavos, cuartos, side) {
  const frag = document.createDocumentFragment();

  // Columna 16avos
  const col16 = makeRoundCol(side==='left' ? '16avos' : '16avos');
  r16Pairs.forEach(([idA, idB]) => {
    const mA = BRACKET_DEF[idA], mB = BRACKET_DEF[idB];
    col16.appendChild(makeMatchGroup(mA.a, mA.b, predIdx, mcIdx, accIdx, winner[idA], 'Round of 32', side));
    col16.appendChild(makeMatchGroup(mB.a, mB.b, predIdx, mcIdx, accIdx, winner[idB], 'Round of 32', side));
  });
  frag.appendChild(col16);

  // Columna 8avos
  const col8 = makeRoundCol('8avos');
  r8Pairs.forEach(([r8keyA, r8keyB]) => {
    // equipos que llegan a este octavo
    const pairs = r16Pairs;
    // teams from projected winners
    const tA = octavos[r8keyA] || '?', tB = octavos[r8keyB] || '?';
    const winner8 = projectedWinner(predIdx, tA, tB);
    col8.appendChild(makeMatchGroup(tA, tB, predIdx, mcIdx, accIdx, winner8, 'Round of 16', side));
  });
  frag.appendChild(col8);

  // Columna cuartos
  const colQ = makeRoundCol('Cuartos');
  qfPairs.forEach(([qA, qB]) => {
    const tA = cuartos[qA] || '?', tB = cuartos[qB] || '?';
    const winQ = projectedWinner(predIdx, tA, tB);
    colQ.appendChild(makeMatchGroup(tA, tB, predIdx, mcIdx, accIdx, winQ, 'Quarter-finals', side));
  });
  frag.appendChild(colQ);

  // Columna semis
  const colS = makeRoundCol('Semis');
  sfTeams.forEach(team => {
    const g = document.createElement('div');
    g.className = 'b-group';
    g.style.flex = '1';
    g.appendChild(makeTeamEl(team, predIdx, mcIdx, 'Semi-finals', true, false));
    colS.appendChild(g);
  });
  frag.appendChild(colS);

  return frag;
}

function makeRoundCol(title) {
  const col = document.createElement('div');
  col.className = 'b-round';
  const t = document.createElement('div');
  t.className = 'b-round-title'; t.textContent = title;
  col.appendChild(t);
  return col;
}

function makeMatchGroup(teamA, teamB, predIdx, mcIdx, accIdx, projectedWin, roundKey, side) {
  const g = document.createElement('div');
  g.className = 'b-group';
  g.appendChild(makeTeamEl(teamA, predIdx, mcIdx, roundKey, teamA===projectedWin, false));
  g.appendChild(makeTeamEl(teamB, predIdx, mcIdx, roundKey, teamB===projectedWin, false));
  return g;
}

function makeTeamEl(name, predIdx, mcIdx, roundKey, isFav, isChamp) {
  const el = document.createElement('div');
  el.className = 'b-team' + (isChamp ? ' champion' : isFav ? ' fav' : '');
  const prob = getRoundProb(mcIdx, name, roundKey);
  const probStr = prob !== null ? prob.toFixed(1)+'%' : '—';
  el.innerHTML = `<span class="b-team__name" title="${name}">${name}</span>
                  <span class="b-team__prob">${probStr}</span>`;
  return el;
}

function buildFinalTeamEl(team, predIdx, mcIdx, roundKey, isChamp) {
  const prob = getRoundProb(mcIdx, team, roundKey);
  return `<div class="b-final-team${isChamp?' champion':''}">
    <span style="font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:90px">${team}</span>
    <span style="color:var(--gold);font-size:.75rem;font-weight:700">${prob!==null?prob.toFixed(1)+'%':'—'}</span>
  </div>`;
}

// placeholder no usado
function buildR16Col(){return document.createElement('div')}

// ══════════════════════════════════════════════════
// MATCH CARDS con resultado real
// ══════════════════════════════════════════════════
function renderMatches(predictions, accuracy) {
  const grid = document.getElementById('matchesGrid');
  grid.innerHTML = '';
  const accIdx = buildAccIdx(accuracy);

  // Ordenar: partidos con resultado real primero, luego por prob desc
  const matches = [...(predictions.matches||[])].sort((a,b) => {
    const ra = accIdx[a.team_a+'|'+a.team_b], rb = accIdx[b.team_a+'|'+b.team_b];
    if (ra && !rb) return -1; if (!ra && rb) return 1;
    return b.prob_a - a.prob_a;
  });

  matches.forEach(m => {
    const pA = (m.prob_a*100).toFixed(1), pB = (m.prob_b*100).toFixed(1);
    const aFav = m.prob_a >= m.prob_b;
    const real = accIdx[m.team_a+'|'+m.team_b];

    let resultHtml = '';
    if (real) {
      const label = real.correct
        ? `✓ Acertado · ${real.real_winner} ganó`
        : `✗ Fallado · Ganó ${real.real_winner}, predijimos ${real.predicted_winner}`;
      resultHtml = `<div class="match-card__result ${real.correct?'correct':'wrong'}">${label}</div>`;
    } else {
      resultHtml = `<div class="match-card__result pending">⏳ Pendiente de jugar</div>`;
    }

    const card = document.createElement('div');
    card.className = 'match-card';
    card.innerHTML = `
      ${resultHtml}
      <div class="match-card__teams">
        <div class="match-card__team">
          <div class="match-card__team-name">${m.team_a}</div>
          <div class="match-card__team-prob${aFav?'':' dim'}">${pA}%</div>
        </div>
        <div class="match-card__vs">VS</div>
        <div class="match-card__team">
          <div class="match-card__team-name">${m.team_b}</div>
          <div class="match-card__team-prob${!aFav?'':' dim'}">${pB}%</div>
        </div>
      </div>
      <div class="prob-bar">
        <div class="prob-bar__a" style="width:${pA}%"></div>
        <div class="prob-bar__b" style="width:${pB}%"></div>
      </div>
      <div class="match-card__signals">
        <div class="chip">Elo <b>${(m.p_elo_a*100).toFixed(0)}%</b></div>
        <div class="chip">FIFA <b>${(m.p_fifa_a*100).toFixed(0)}%</b></div>
        <div class="chip">H2H <b>${m.h2h_used?(m.p_h2h_a*100).toFixed(0)+'%':'N/A'}</b></div>
        <div class="chip clickable" onclick="openH2H('${m.team_a}','${m.team_b}',${m.h2h_matches})">
          📋 ${m.h2h_matches} partidos hist.
        </div>
      </div>
      <div class="match-card__foot">
        <span class="badge ${m.confidence}">${m.confidence} confianza</span>
      </div>`;
    grid.appendChild(card);
  });
}

// ══════════════════════════════════════════════════
// MODAL H2H
// ══════════════════════════════════════════════════
async function openH2H(teamA, teamB, totalMatches) {
  document.getElementById('modalTitle').textContent = teamA + ' vs ' + teamB;
  document.getElementById('modalOverlay').classList.add('open');

  const body = document.getElementById('modalBody');
  body.innerHTML = '<p style="color:var(--muted);font-size:.85rem">Cargando historial…</p>';

  try {
    const data = await fetchJSON(`/api/h2h/${encodeURIComponent(teamA)}/${encodeURIComponent(teamB)}`);
    renderModalH2H(body, data, teamA, teamB);
  } catch(e) {
    body.innerHTML = `<p class="h2h-empty">No se pudo cargar el historial de enfrentamientos.</p>`;
  }
}

function renderModalH2H(body, data, teamA, teamB) {
  const wA = (data.win_rate_a*100).toFixed(0);
  const wB = (data.win_rate_b*100).toFixed(0);
  const dr = (data.draw_rate*100).toFixed(0);

  body.innerHTML = `
    <div class="h2h-summary">
      <div class="h2h-stat"><span class="h2h-stat__n">${data.total_matches}</span><span class="h2h-stat__l">Partidos</span></div>
      <div class="h2h-stat"><span class="h2h-stat__n" style="color:var(--green)">${wA}%</span><span class="h2h-stat__l">${teamA}</span></div>
      <div class="h2h-stat"><span class="h2h-stat__n">${dr}%</span><span class="h2h-stat__l">Empates</span></div>
      <div class="h2h-stat"><span class="h2h-stat__n" style="color:var(--red)">${wB}%</span><span class="h2h-stat__l">${teamB}</span></div>
      <div class="h2h-stat"><span class="h2h-stat__n">${data.goal_diff_a>0?'+':''}${data.goal_diff_a}</span><span class="h2h-stat__l">Dif. goles</span></div>
    </div>
    <div class="h2h-matches-title">Últimos enfrentamientos</div>
    ${data.total_matches === 0
      ? `<div class="h2h-empty">Nunca se han enfrentado antes de este torneo</div>`
      : (data.last_5||[]).map(m => `
          <div class="h2h-match">
            <span class="h2h-match__date">${m.date}</span>
            <span class="h2h-match__score">${m.score}</span>
            <span class="h2h-match__tourn">${m.tournament}</span>
          </div>`).join('')
    }`;
}

function closeModal() {
  document.getElementById('modalOverlay').classList.remove('open');
}
document.addEventListener('keydown', e => { if(e.key==='Escape') closeModal(); });

// ══════════════════════════════════════════════════
// HEATMAP
// ══════════════════════════════════════════════════
function renderHeatmap(mc) {
  const table = document.getElementById('heatmapTable');
  table.innerHTML = '';
  const labels  = mc.round_labels||[];
  const allKeys = ROUND_KEYS.filter(k => labels.includes(ROUND_ES[k])||labels.includes(k));

  const thead = document.createElement('thead');
  const hr    = document.createElement('tr');
  ['Selección','Índice fuerza',...labels].forEach((l,i) => {
    const th = document.createElement('th');
    th.textContent=l; if(i===0) th.className='th-team'; hr.appendChild(th);
  });
  thead.appendChild(hr); table.appendChild(thead);

  const tbody  = document.createElement('tbody');
  const sorted = [...(mc.teams||[])].sort((a,b)=>(b['Final']||0)-(a['Final']||0));
  sorted.forEach(t => {
    const tr = document.createElement('tr');
    const tdN = document.createElement('td'); tdN.className='td-team'; tdN.textContent=t.team; tr.appendChild(tdN);
    const tdE = document.createElement('td'); tdE.className='td-elo'; tdE.textContent=t.elo?t.elo.toFixed(0):'—'; tr.appendChild(tdE);
    allKeys.forEach(k => {
      const v=t[k]!==undefined?t[k]:0;
      const td=document.createElement('td'); td.textContent=v.toFixed(1)+'%'; td.className=heatCls(v); tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
}

function heatCls(v){
  if(v===0)return 'h0';if(v<10)return 'h1';if(v<25)return 'h2';
  if(v<45)return 'h3';if(v<65)return 'h4';if(v<85)return 'h5';return 'h6';
}

// ══════════════════════════════════════════════════
// ELO CHART
// ══════════════════════════════════════════════════
function renderElo(elo) {
  const teams = [...(elo.teams||[])].sort((a,b)=>b.elo-a.elo);
  const labels = teams.map(t=>t.team), vals = teams.map(t=>t.elo);
  const max=Math.max(...vals), min=Math.min(...vals);
  const colors = vals.map(v=>{
    const r=(v-min)/(max-min);
    if(r>.75)return 'rgba(240,180,41,.9)';if(r>.5)return 'rgba(249,115,22,.8)';
    if(r>.25)return 'rgba(59,130,246,.8)';return 'rgba(30,58,95,.9)';
  });
  const ctx=document.getElementById('eloChart').getContext('2d');
  if(eloChart) eloChart.destroy();
  eloChart=new Chart(ctx,{
    type:'bar',data:{labels,datasets:[{label:'Índice de fuerza',data:vals,backgroundColor:colors,borderColor:'transparent',borderRadius:4,barThickness:13}]},
    options:{indexAxis:'y',responsive:true,maintainAspectRatio:false,
      plugins:{legend:{display:false},tooltip:{callbacks:{label:c=>' Fuerza: '+c.raw.toFixed(1)},backgroundColor:'#161f30',borderColor:'#1e2d45',borderWidth:1,titleColor:'#e8edf5',bodyColor:'#f0b429'}},
      scales:{x:{min:min-60,grid:{color:'#1e2d45'},ticks:{color:'#6b7a99',font:{family:'Inter',size:11}}},y:{grid:{display:false},ticks:{color:'#e8edf5',font:{family:'Inter',size:11}}}}}
  });
  document.querySelector('.elo-chart-box').style.height=(teams.length*27+60)+'px';
}

// ══════════════════════════════════════════════════
// ACCURACY
// ══════════════════════════════════════════════════
function renderAccuracy(acc) {
  const pct=acc.overall_accuracy||0;
  document.getElementById('dialPct').textContent=pct.toFixed(1)+'%';
  document.getElementById('dialSub').textContent=(acc.correct||0)+' / '+(acc.total_matches||0)+' partidos';
  setTimeout(()=>{document.getElementById('dialFill').style.strokeDashoffset=251-(pct/100)*251;},300);

  const table=document.getElementById('accuracyTable'); table.innerHTML='';
  const thead=document.createElement('thead');
  thead.innerHTML='<tr><th>Partido</th><th>Ronda</th><th>Predicción</th><th>Resultado real</th><th>Prob.</th><th>¿Acertado?</th></tr>';
  table.appendChild(thead);
  const tbody=document.createElement('tbody');
  const results=acc.match_results||[];
  if(!results.length){
    const tr=document.createElement('tr');
    tr.innerHTML='<td colspan="6" class="empty-msg">Los resultados aparecerán aquí en cuanto se jueguen los partidos</td>';
    tbody.appendChild(tr);
  } else {
    results.forEach(r=>{
      const tr=document.createElement('tr');
      const probFav=Math.max(r.prob_a,r.prob_b);
      tr.innerHTML=`<td>${r.team_a} vs ${r.team_b}</td><td>${r.round}</td><td>${r.predicted_winner}</td><td>${r.real_winner}</td><td>${(probFav*100).toFixed(1)}%</td><td class="${r.correct?'ok':'ko'}">${r.correct?'✓ Sí':'✗ No'}</td>`;
      tbody.appendChild(tr);
    });
  }
  table.appendChild(tbody);
}