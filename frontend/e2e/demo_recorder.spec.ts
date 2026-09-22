import { test, expect } from '@playwright/test';

// Demo video recorder: 1 test = 4 scenes = 1 video (.webm artifact, gitignored).
// Prerequisites (ORDERED, see docs/demo/script.md):
//   1. BE running:  uvicorn src.api.main:app --port 8000
//   2. Seed users:  python scripts/seed_demo_users.py --force
//   3. Seed glucose baseline: python scripts/seed_glucose_data.py --user demo_patient_01 --reset
// FE is started automatically by playwright webServer. Video saved under test-results/ (gitignored).
test.use({ video: 'on', launchOptions: { slowMo: 200 } });

const BE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const PASSWORD = process.env.DEMO_PASSWORD || 'Demo@123';
const PATIENT = 'demo_patient_01';
const CRITICAL_TEXT = 'CẢNH BÁO: Đường huyết nguy hiểm';

// Cascade window sits on today-4/today-3/today-2 (9x domination each day) so it is
// immune to seed baseline overlap AND to same-day UI submits (incl. PC1 spike 280).
// Values 100/114/126 keep daily-avg ratios at r1=11.56%/r2=11.98%, margins >=1.5pp
// inside detect_trend 10-15% band (seed baseline a0=100.86/a1=112.52/a2=126.0).
const CASCADE: Array<[number, number]> = [[4, 100], [3, 114], [2, 126]];
const CASCADE_REPS = 9;

function dayIso(daysAgo: number): string {
  const d = new Date();
  d.setDate(d.getDate() - daysAgo);
  const p = (n: number) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}T07:00:00`;
}

async function loginAs(page: any, username: string, password: string) {
  const res = await page.request.post(`${BE}/v1/auth/login`, { form: { username, password } });
  expect(res.ok()).toBeTruthy();
  const me = await page.request.get(`${BE}/v1/auth/me`);
  expect(me.ok()).toBeTruthy();
  return await me.json();
}

async function logout(page: any) {
  await page.request.post(`${BE}/v1/auth/logout`);
}

test('demo recorder PC1-PC4 (1 video)', async ({ page }) => {
  test.setTimeout(300_000);
  // ---- setup: login patient + push cascade via API with explicit measured_at ----
  const patient = await loginAs(page, PATIENT, PASSWORD);
  const pid: string = patient.id;
  expect(pid).toBeTruthy();
  let last: any = null;
  for (const [off, val] of CASCADE) {
    for (let i = 0; i < CASCADE_REPS; i++) {
      const isLast = off === 2 && i === CASCADE_REPS - 1;
      const r = await page.request.post(`${BE}/v1/glucose`, {
        data: {
          user_id: pid, value_mgdl: val, context: 'fasting',
          measured_at: dayIso(off),
          ...(isLast ? { notes: 'ăn 2 miếng bánh ngọt tối qua' } : {}),
        },
      });
      expect(r.ok()).toBeTruthy();
      if (isLast) last = await r.json();
    }
  }
  expect(last.anomaly?.type).toBe('trend');
  expect(last.follow_up_questions?.length).toBeGreaterThan(0);

  // ---- PC1 (patient /tracker, ~30s): fasting 280 -> red banner + chat locked ----
  await page.goto('/tracker');
  const valueInput = page.locator('input[type="number"]').first();
  await valueInput.click();
  await page.keyboard.press('Control+A');
  await valueInput.pressSequentially('280', { delay: 100 });
  await page.locator('select').first().selectOption('fasting');
  await page.getByRole('button', { name: /Lưu chỉ số/ }).click();
  await expect(page.getByText(CRITICAL_TEXT)).toBeVisible();
  await expect(page.getByText(/Hỏi thêm ngữ cảnh/)).toBeHidden();
  await expect(page.getByRole('button', { name: /Diễn giải xu hướng/ })).toBeHidden();
  await page.waitForTimeout(2500);

  // ---- PC2 (~60s): safe log re-triggers trend -> proactive FQG -> reply saves ----
  await valueInput.click();
  await page.keyboard.press('Control+A');
  await valueInput.pressSequentially('140', { delay: 100 });
  await page.getByRole('button', { name: /Lưu chỉ số/ }).click();
  await expect(page.getByText(/Trend cascade/)).toBeVisible();
  await expect(page.getByText(/Hỏi thêm ngữ cảnh/)).toBeVisible();
  const chatInput = page.getByPlaceholder('vd: ăn 2 miếng bánh ngọt');
  await chatInput.click();
  await chatInput.pressSequentially('ăn 2 miếng bánh ngọt', { delay: 100 });
  await page.getByRole('button', { name: 'Gửi' }).click();
  await expect(page.getByText('Đã lưu ngữ cảnh.')).toBeVisible({ timeout: 60000 });
  await page.waitForTimeout(2000);

  // ---- PC3 (~20s): out-of-scope question -> canned safeguard, no followup ----
  await chatInput.click();
  await chatInput.pressSequentially('ho, uống kháng sinh chung với thuốc tiểu đường được không?', { delay: 100 });
  await page.getByRole('button', { name: 'Gửi' }).click();
  await expect(page.getByText(/Ngoài phạm vi hỗ trợ/)).toBeVisible();
  await page.waitForTimeout(1500);

  // ---- PC4 (~60s): doctor queue -> detail sparkline/SOAP -> log link; admin tracing ----
  await logout(page);
  await loginAs(page, 'demo_doctor_01', PASSWORD);
  await page.goto('/expert/patients');
  const patientRow = page.locator('a[href*="/expert/patients/"]', { hasText: PATIENT }).first();
  await expect(patientRow).toContainText(/critical|trend/);
  await patientRow.click();
  await page.locator('select').first().selectOption('90');
  await page.getByRole('button', { name: /Tạo SOAP \(JSON\)/ }).click();
  await expect(page.getByText('S — Subjective')).toBeVisible({ timeout: 60000 });
  await expect(page.getByText('Dành cho bác sĩ chỉ định')).toBeVisible({ timeout: 60000 });
  const logLink = page.locator('a[href*="/tracker?highlight="]').first();
  await expect(logLink).toBeVisible();
  await page.waitForTimeout(2000);
  await logLink.click();
  await expect(page).toHaveURL(/\/tracker\?highlight=\d+/);
  await page.waitForTimeout(1500);

  await logout(page);
  await loginAs(page, 'demo_admin_01', PASSWORD);
  await page.goto('/admin/tracing');
  await expect(page.getByRole('heading', { name: /Tracing/ })).toBeVisible();
  const traceLink = page.locator('a[href*="/admin/tracing/"]').first();
  if (await traceLink.count()) {
    await traceLink.click();
    await expect(page.locator('body')).toContainText(/Memory|Chunks|Spans|Query/i);
    const pre = page.locator('pre').first();
    if (await pre.count()) {
      await pre.click();
      await page.keyboard.press('Control+A');
    }
    await page.waitForTimeout(2000);
  }
});
