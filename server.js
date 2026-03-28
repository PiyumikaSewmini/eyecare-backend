const express    = require('express');
const cors       = require('cors');
const multer     = require('multer');
const path       = require('path');
const { spawn }  = require('child_process');
const fs         = require('fs');
const bcrypt     = require('bcrypt');
const jwt        = require('jsonwebtoken');
const mysql      = require('mysql2/promise');

const app        = express();
const PORT       = 5000;
const JWT_SECRET = process.env.JWT_SECRET || 'your-secret-key-change-in-production';

const pool = mysql.createPool({
  host: process.env.DB_HOST || 'localhost', user: process.env.DB_USER || 'root',
  password: process.env.DB_PASSWORD || '', database: process.env.DB_NAME || 'dr_screening',
  waitForConnections: true, connectionLimit: 10, queueLimit: 0,
});

// ──FIX: Explicit CORS config that allows DELETE method ────────────────────
app.use(cors({
  origin: '*',
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization'],
}));

// Handle OPTIONS preflight for all routes (regex avoids path-to-regexp crash)
app.options(/.*/, cors());

app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));

// ── Multer ────────────────────────────────────────────────────────────────────
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const dir = path.join(__dirname, 'uploads');
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
    cb(null, dir);
  },
  filename: (req, file, cb) => {
    cb(null, `fundus-${Date.now()}-${Math.random().toString(36).substr(2, 9)}-${file.originalname}`);
  },
});
const upload = multer({
  storage, limits: { fileSize: 10 * 1024 * 1024 },
  fileFilter: (req, file, cb) => {
    const ok = /jpeg|jpg|png/.test(path.extname(file.originalname).toLowerCase()) && /jpeg|jpg|png/.test(file.mimetype);
    ok ? cb(null, true) : cb(new Error('Only JPEG/JPG/PNG files are allowed'));
  },
});

// ── Helpers ───────────────────────────────────────────────────────────────────
function safeUnlink(filePath) {
  try { if (filePath && fs.existsSync(filePath)) fs.unlinkSync(filePath); } catch (_) {}
}

function validateClinicalData(data) {
  const errors = [];
  if (!data.age   || data.age   < 18 || data.age   > 120) errors.push('Age must be between 18 and 120');
  if (!data.hba1c || data.hba1c < 4  || data.hba1c > 15)  errors.push('HbA1c must be between 4 and 15%');
  if (!data.diabetes_duration && data.diabetes_duration !== 0)
    errors.push('Diabetes duration is required');
  if (!data.systolic_bp  || data.systolic_bp  < 60 || data.systolic_bp  > 300)
    errors.push('Systolic BP must be between 60 and 300 mmHg');
  if (!data.diastolic_bp || data.diastolic_bp < 30 || data.diastolic_bp > 200)
    errors.push('Diastolic BP must be between 30 and 200 mmHg');
  return errors;
}

function runImageValidation(imagePath) {
  return new Promise((resolve) => {
    const script = path.join(__dirname, '..', 'DR_Training', 'validate_image.py');
    if (!fs.existsSync(script)) {
      resolve({ valid: true, score: 70, message: 'Validation skipped (script not found)' });
      return;
    }
    const proc = spawn('python', [script, imagePath]);
    let stdout = '', stderr = '';
    proc.stdout.on('data', d => { stdout += d.toString(); });
    proc.stderr.on('data', d => { stderr += d.toString(); });
    proc.on('close', () => {
      try { resolve(JSON.parse(stdout.trim())); }
      catch (_) { resolve({ valid: true, score: 50, message: 'Validation inconclusive' }); }
    });
    setTimeout(() => { proc.kill(); resolve({ valid: true, score: 50, message: 'Validation timed out' }); }, 12000);
  });
}

// ── Auth middleware ───────────────────────────────────────────────────────────
const auth = (req, res, next) => {
  const token = (req.headers['authorization'] || '').split(' ')[1];
  if (!token) return res.status(401).json({ error: 'ACCESS_DENIED' });
  jwt.verify(token, JWT_SECRET, (err, user) => {
    if (err) {
      console.error('JWT verify error:', err.message);
      return res.status(403).json({ error: 'INVALID_TOKEN' });
    }
    req.user = user; next();
  });
};

// =============================================================================
// Health
// =============================================================================
app.get('/health', (_, res) => res.json({ status: 'OK', timestamp: new Date().toISOString() }));

// =============================================================================
// Auth
// =============================================================================
app.post('/api/auth/register', async (req, res) => {
  try {
    const { email, password, fullName, dateOfBirth, phone } = req.body;
    if (!email || !password || !fullName)
      return res.status(400).json({ error: 'MISSING_FIELDS', message: 'Email, password, and full name are required' });
    const [existing] = await pool.query('SELECT id FROM users WHERE email = ?', [email]);
    if (existing.length > 0) return res.status(400).json({ error: 'USER_EXISTS', message: 'Email already registered' });
    const hash = await bcrypt.hash(password, 10);
    const [result] = await pool.query(
      'INSERT INTO users (email, password_hash, full_name, date_of_birth, phone) VALUES (?, ?, ?, ?, ?)',
      [email, hash, fullName, dateOfBirth || null, phone || null]
    );
    await pool.query('INSERT INTO patient_profiles (user_id) VALUES (?)', [result.insertId]);
    const token = jwt.sign({ userId: result.insertId, email }, JWT_SECRET, { expiresIn: '7d' });
    res.json({ success: true, token, user: { id: result.insertId, email, fullName } });
  } catch (e) { res.status(500).json({ error: 'SERVER_ERROR', message: 'Registration failed' }); }
});

app.post('/api/auth/login', async (req, res) => {
  try {
    const { email, password } = req.body;
    if (!email || !password) return res.status(400).json({ error: 'MISSING_FIELDS' });
    const [users] = await pool.query('SELECT * FROM users WHERE email = ? AND is_active = TRUE', [email]);
    if (!users.length) return res.status(401).json({ error: 'INVALID_CREDENTIALS', message: 'Invalid email or password' });
    const user = users[0];
    if (!(await bcrypt.compare(password, user.password_hash)))
      return res.status(401).json({ error: 'INVALID_CREDENTIALS', message: 'Invalid email or password' });
    await pool.query('UPDATE users SET last_login = NOW() WHERE id = ?', [user.id]);
    const token = jwt.sign({ userId: user.id, email: user.email }, JWT_SECRET, { expiresIn: '7d' });
    res.json({ success: true, token, user: { id: user.id, email: user.email, fullName: user.full_name } });
  } catch (e) { res.status(500).json({ error: 'SERVER_ERROR', message: 'Login failed' }); }
});

app.get('/api/auth/me', auth, async (req, res) => {
  try {
    const [users] = await pool.query(
      'SELECT id, email, full_name, date_of_birth, phone, created_at FROM users WHERE id = ?', [req.user.userId]
    );
    if (!users.length) return res.status(404).json({ error: 'USER_NOT_FOUND' });
    res.json({ user: users[0] });
  } catch (e) { res.status(500).json({ error: 'SERVER_ERROR' }); }
});

// =============================================================================
// Diagnose
// =============================================================================
app.post('/api/diagnose', auth, upload.single('fundusImage'), async (req, res) => {
  const start = Date.now();
  let imagePath = null;
  try {
    if (!req.file) return res.status(400).json({ error: 'NO_IMAGE', message: 'Please upload a retinal image' });
    imagePath = req.file.path;

    const validation = await runImageValidation(imagePath);
    if (!validation.valid) {
      safeUnlink(imagePath);
      return res.status(400).json({ error: 'INVALID_IMAGE', message: validation.message, validationScore: validation.score, canProceed: false });
    }

    const clinical = {
      age:               parseInt(req.body.age)               || 0,
      hba1c:             parseFloat(req.body.hba1c)           || 0,
      systolic_bp:       parseInt(req.body.systolic_bp)       || 0,
      diastolic_bp:      parseInt(req.body.diastolic_bp)      || 0,
      diabetes_duration: parseInt(req.body.diabetes_duration) || 0,
      bmi:               parseFloat(req.body.bmi)             || 0,
      fasting_glucose:   parseInt(req.body.fasting_glucose)   || 0,
      cholesterol:       parseInt(req.body.cholesterol)       || 0,
    };
    const clinicalErrors = validateClinicalData(clinical);
    if (clinicalErrors.length) {
      safeUnlink(imagePath);
      return res.status(400).json({ error: 'INVALID_DATA', message: clinicalErrors.join(', '), errors: clinicalErrors });
    }

    const scriptNames = ['predict_xai.py', 'predict_working.py', 'predict.py'];
    let script = null;
    for (const s of scriptNames) {
      const p = path.join(__dirname, '..', 'DR_Training', s);
      if (fs.existsSync(p)) { script = p; break; }
    }
    if (!script) {
      safeUnlink(imagePath);
      return res.status(500).json({ error: 'MODEL_NOT_FOUND', message: 'AI model scripts not found on server' });
    }

    const aiProcess = spawn('python', [script, imagePath, JSON.stringify(clinical)]);
    let stdout = '', stderr = '';
    aiProcess.stdout.on('data', d => { stdout += d.toString(); });
    aiProcess.stderr.on('data', d => { stderr += d.toString(); });

    const killTimer = setTimeout(() => {
      if (aiProcess.exitCode === null) {
        aiProcess.kill(); safeUnlink(imagePath);
        res.status(408).json({ error: 'TIMEOUT', message: 'Analysis took too long. Please try again.' });
      }
    }, 90000);

    aiProcess.on('close', async (code) => {
      clearTimeout(killTimer);
      const elapsed = ((Date.now() - start) / 1000).toFixed(1);
      if (code !== 0) { safeUnlink(imagePath); return res.status(500).json({ error: 'PREDICTION_FAILED', details: stderr }); }
      try {
        const result = JSON.parse(stdout.trim());
        if (result.error === 'INVALID_IMAGE') { safeUnlink(imagePath); return res.status(400).json({ ...result, validationScore: result.validationScore || 0, canProceed: false }); }
        if (result.error) { safeUnlink(imagePath); return res.status(400).json(result); }

        const relPath = path.relative(path.join(__dirname, 'uploads'), imagePath);
        await pool.query(`
          INSERT INTO screening_history
            (user_id, age, hba1c, systolic_bp, diastolic_bp, bmi, diabetes_duration,
             fasting_glucose, cholesterol, severity, stage, confidence, risk_score,
             image_path, probabilities, recommendations)
          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        `, [
          req.user.userId,
          clinical.age, clinical.hba1c, clinical.systolic_bp, clinical.diastolic_bp,
          clinical.bmi, clinical.diabetes_duration, clinical.fasting_glucose, clinical.cholesterol,
          result.severity, result.stage, result.confidence,
          result.riskScore || result.risk_score || 0,
          relPath,
          JSON.stringify(result.probabilities   || {}),
          JSON.stringify(result.recommendations || {}),
        ]);

        res.json({ ...result, saved: true, processingTime: parseFloat(elapsed), timestamp: new Date().toISOString() });
      } catch (e) {
        safeUnlink(imagePath);
        res.status(500).json({ error: 'PROCESSING_ERROR', message: 'Failed to process AI results' });
      }
    });
  } catch (err) {
    safeUnlink(imagePath);
    res.status(500).json({ error: 'SERVER_ERROR', message: err.message });
  }
});

// =============================================================================
// History — GET list, GET single, DELETE
// =============================================================================
app.get('/api/history', auth, async (req, res) => {
  try {
    const [history] = await pool.query(
      `SELECT id, screening_date, severity, stage, confidence, risk_score,
              age, hba1c, systolic_bp, diastolic_bp, diabetes_duration,
              bmi, fasting_glucose, cholesterol, image_path, probabilities
       FROM screening_history WHERE user_id = ?
       ORDER BY screening_date DESC LIMIT 50`,
      [req.user.userId]
    );
    history.forEach(row => {
      try { row.probabilities = JSON.parse(row.probabilities || '{}'); } catch { row.probabilities = {}; }
    });
    res.json({ history });
  } catch (e) { res.status(500).json({ error: 'SERVER_ERROR' }); }
});

app.get('/api/history/:id', auth, async (req, res) => {
  try {
    const [rows] = await pool.query(
      'SELECT * FROM screening_history WHERE id = ? AND user_id = ?',
      [req.params.id, req.user.userId]
    );
    if (!rows.length) return res.status(404).json({ error: 'NOT_FOUND' });
    const s = rows[0];
    try { s.probabilities   = JSON.parse(s.probabilities   || '{}'); } catch { s.probabilities = {}; }
    try { s.recommendations = JSON.parse(s.recommendations || '{}'); } catch { s.recommendations = {}; }
    res.json({ screening: s });
  } catch (e) { res.status(500).json({ error: 'SERVER_ERROR' }); }
});

/**
 * DELETE /api/history/:id
 * Fixed: explicit logging, no parseInt (mysql2 handles coercion),
 *    user_id matched directly from JWT.
 */
app.delete('/api/history/:id', auth, async (req, res) => {
  try {
    const recordId = req.params.id;         // keep as string — mysql2 handles it
    const userId   = req.user.userId;       // from JWT payload

    console.log(` DELETE /api/history/${recordId} — requested by user ${userId}`);

    //  First, try fetching WITHOUT user_id to distinguish "not found" vs "not yours"
    const [any] = await pool.query(
      'SELECT id, user_id, image_path FROM screening_history WHERE id = ?',
      [recordId]
    );

    if (!any.length) {
      console.warn(`   Record ${recordId} does not exist in DB`);
      return res.status(404).json({ error: 'NOT_FOUND', message: 'Record not found' });
    }

    // Check ownership separately for a clearer error
    // Coerce both to Number for safe comparison (handles BigInt/string edge cases)
    if (Number(any[0].user_id) !== Number(userId)) {
      console.warn(`   Record ${recordId} belongs to user ${any[0].user_id}, not ${userId}`);
      return res.status(403).json({ error: 'FORBIDDEN', message: 'You do not own this record' });
    }

    const record = any[0];

    // Delete from database
    await pool.query('DELETE FROM screening_history WHERE id = ?', [recordId]);

    // Delete image file from disk
    if (record.image_path) {
      const fullImagePath = path.join(__dirname, 'uploads', record.image_path);
      safeUnlink(fullImagePath);
      const heatmapPath = fullImagePath.replace(/\.(jpe?g|png)$/i, '_heatmap.$1');
      safeUnlink(heatmapPath);
    }

    console.log(`Record ${recordId} deleted successfully`);
    res.json({ success: true, message: 'Record permanently deleted', deletedId: Number(recordId) });

  } catch (e) {
    console.error('Delete error:', e);
    res.status(500).json({ error: 'SERVER_ERROR', message: e.message || 'Failed to delete record' });
  }
});

// =============================================================================
// Dashboard stats
// =============================================================================
app.get('/api/dashboard/stats', auth, async (req, res) => {
  try {
    const [[{ count }]] = await pool.query('SELECT COUNT(*) as count FROM screening_history WHERE user_id = ?', [req.user.userId]);
    const [latest]      = await pool.query('SELECT severity, stage, screening_date, risk_score FROM screening_history WHERE user_id = ? ORDER BY screening_date DESC LIMIT 1', [req.user.userId]);
    res.json({ totalScreenings: count, latestScreening: latest[0] || null });
  } catch (e) { res.status(500).json({ error: 'SERVER_ERROR' }); }
});

// =============================================================================
// Profile
// =============================================================================
app.put('/api/profile', auth, async (req, res) => {
  try {
    const { fullName, dateOfBirth, phone, diabetesType, medications, allergies } = req.body;
    await pool.query('UPDATE users SET full_name=?, date_of_birth=?, phone=? WHERE id=?', [fullName, dateOfBirth, phone, req.user.userId]);
    await pool.query('UPDATE patient_profiles SET diabetes_type=?, current_medications=?, allergies=? WHERE user_id=?', [diabetesType, medications, allergies, req.user.userId]);
    res.json({ success: true });
  } catch (e) { res.status(500).json({ error: 'SERVER_ERROR' }); }
});

// =============================================================================
// Error handler
// =============================================================================
app.use((err, req, res, next) => {
  if (err instanceof multer.MulterError && err.code === 'LIMIT_FILE_SIZE')
    return res.status(400).json({ error: 'FILE_TOO_LARGE', message: 'Maximum file size is 10 MB' });
  res.status(500).json({ error: 'SERVER_ERROR', message: err.message });
});

// =============================================================================
// Start
// =============================================================================
app.listen(PORT, () => {
  console.log('\n' + '='.repeat(70));
  console.log('DR SCREENING BACKEND');
  console.log('='.repeat(70));
  console.log(`Server  : http://localhost:${PORT}`);
  console.log('\nEndpoints:');
  [
    'POST /api/auth/register',
    'POST /api/auth/login',
    'POST /api/diagnose (auth)',
    'GET  /api/history (auth)',
    'GET  /api/history/:id (auth)',
    'DELETE /api/history/:id (auth)',   // ← must appear in this list
    'GET  /api/dashboard/stats (auth)',
  ].forEach(e => console.log(`  ${e}`));
  console.log('='.repeat(70) + '\n');
});

module.exports = app;