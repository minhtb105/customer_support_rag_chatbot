import { test, expect } from './fixtures';

// Guided UI (auth via useAuth, no User ID field): session mocked through /v1/auth/me.
// BE is not started by webServer, so API is mocked to match current UI contracts.
const ME = { id: 'e2e_user', username: 'e2e_user', role: 'user', is_active: true, is_verified: true };

function dayIso(): string {
  const d = new Date();
  const p = (n: number) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}T07:00:00`;
}

function mockSession(page: any) {
  return page.route('**/v1/auth/me', async (route: any) => {
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(ME) });
  });
}

function mockGlucoseGet(page: any, logs: any[]) {
  return page.route('**/v1/glucose/*', async (route: any) => {
    if (route.request().method() !== 'GET') {
      await route.continue();
      return;
    }
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        user_id: ME.id,
        logs,
        stats: { user_id: ME.id, total_logs: logs.length, avg_mgdl: 130, last_7_days_avg: 130, streak_days: 1, logs_per_week: 0.5, classification_counts: { high: logs.length } },
        should_escalate: false,
        anomaly_ids: [],
      }),
    });
  });
}

function mockGlucosePost(page: any, resp: any) {
  return page.route('**/v1/glucose', async (route: any) => {
    if (route.request().method() !== 'POST') {
      await route.continue();
      return;
    }
    const body = await route.request().postDataJSON();
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        id: 1, user_id: ME.id, value_mgdl: body.value_mgdl, measured_at: dayIso(),
        context: body.context, notes: body.notes, classification: 'high', message: 'high message',
        ...resp,
      }),
    });
  });
}

test.describe('Tracker A — Guided UI', () => {
  test('submit safe log shows KPIs and history', async ({ page }) => {
    await mockSession(page);
    await mockGlucosePost(page, { anomaly: { type: 'none', reason: '' }, follow_up_questions: [] });
    await mockGlucoseGet(page, [
      { id: 1, user_id: ME.id, value_mgdl: 130, measured_at: dayIso(), context: 'fasting', notes: null, classification: 'high' },
    ]);

    await page.goto('/tracker');
    await page.locator('input[type="number"]').first().fill('130');
    await page.locator('select').first().selectOption('fasting');
    await page.getByRole('button', { name: /Lưu chỉ số/ }).click();
    await expect(page.getByText('HIGH: high message')).toBeVisible({ timeout: 5000 });
    await expect(page.getByText(/Tổng logs/)).toBeVisible();
  });

  test('spike submit shows red banner and locks chat', async ({ page }) => {
    await mockSession(page);
    await mockGlucosePost(page, {
      classification: 'high',
      anomaly: { type: 'spike', direction: 'high', reason: 'spike test' },
      follow_up_questions: ['q1?'],
    });
    await mockGlucoseGet(page, []);

    await page.goto('/tracker');
    await page.locator('input[type="number"]').first().fill('280');
    await page.getByRole('button', { name: /Lưu chỉ số/ }).click();
    await expect(page.getByText(/CẢNH BÁO: Đường huyết nguy hiểm/)).toBeVisible({ timeout: 5000 });
    await expect(page.getByTestId('critical-banner')).toBeVisible();
    await expect(page.getByTestId('tracker-save')).toBeVisible();
    await expect(page.getByText(/Hỏi thêm ngữ cảnh/)).toBeHidden();
    await expect(page.getByRole('button', { name: /Diễn giải xu hướng/ })).toBeHidden();
  });

  test('trend submit opens FQG chat and reply saves context', async ({ page }) => {
    await mockSession(page);
    await mockGlucosePost(page, {
      anomaly: { type: 'trend', direction: 'up', reason: 'cascade test' },
      follow_up_questions: ['Khẩu phần ăn gần đây thế nào?'],
    });
    await mockGlucoseGet(page, []);
    await page.route('**/v1/query', async (route: any) => {
      await route.fulfill({
        status: 200, contentType: 'application/json',
        body: JSON.stringify({ answer: 'Ghi nhận ngữ cảnh, tiếp tục theo dõi.', contexts: [] }),
      });
    });
    await page.route('**/v1/glucose/followup', async (route: any) => {
      await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ saved: true }) });
    });

    await page.goto('/tracker');
    await page.locator('input[type="number"]').first().fill('140');
    await page.getByRole('button', { name: /Lưu chỉ số/ }).click();
    await expect(page.getByText(/Cảnh báo chuỗi tăng/)).toBeVisible({ timeout: 5000 });
    await expect(page.getByText(/Hỏi thêm ngữ cảnh/)).toBeVisible();
    await expect(page.getByTestId('fqg-input')).toBeVisible();
    await page.getByPlaceholder('vd: ăn 2 miếng bánh ngọt').fill('ăn 2 miếng bánh ngọt');
    await expect(page.getByTestId('fqg-send')).toBeVisible();
    await page.getByRole('button', { name: 'Gửi' }).click();
    await expect(page.getByText('Đã lưu ngữ cảnh.')).toBeVisible({ timeout: 5000 });
  });

  test('explain trend calls RAG and shows answer', async ({ page }) => {
    await mockSession(page);
    await mockGlucoseGet(page, [
      { id: 1, user_id: ME.id, value_mgdl: 130, measured_at: dayIso(), context: 'fasting', notes: null, classification: 'high' },
    ]);
    await page.route('**/v1/query', async (route: any) => {
      await route.fulfill({
        status: 200, contentType: 'application/json',
        body: JSON.stringify({ answer: 'Xu hướng tăng nhẹ, khuyến nghị theo dõi WHO.', contexts: [] }),
      });
    });

    await page.goto('/tracker');
    await expect(page.getByText(/Biểu đồ 14 ngày/)).toBeVisible();
    await expect(page.getByTestId('trend-explain')).toBeVisible();
    await page.getByRole('button', { name: /Diễn giải xu hướng/ }).click();
    await expect(page.getByText(/Xu hướng tăng nhẹ/)).toBeVisible({ timeout: 5000 });
  });

  test('voice mic transcribes ambiguous then auto-sends via existing path', async ({ page }) => {
    await page.addInitScript(() => {
      const g = navigator as any;
      if (!g.mediaDevices) (g as any).mediaDevices = {};
      (g.mediaDevices as any).getUserMedia = async () => ({ getTracks: () => [] });
      (window as any).MediaRecorder = class {
        ondataavailable: any = null;
        onstop: any = null;
        stream: any;
        constructor(stream: any) { this.stream = stream; }
        start() {}
        stop() { if (this.onstop) this.onstop(); }
      };
    });
    await mockSession(page);
    await mockGlucosePost(page, {
      anomaly: { type: 'trend', direction: 'up', reason: 'cascade test' },
      follow_up_questions: ['Khẩu phần ăn gần đây thế nào?'],
    });
    await mockGlucoseGet(page, []);
    await page.route('**/v1/voice/transcribe', async (route: any) => {
      await route.fulfill({
        status: 200, contentType: 'application/json',
        body: JSON.stringify({ text: 'ăn 2 miếng bánh ngọt', emergency: false, parse: { value: null, context: null, ambiguous: true, reason: 'no_number', confirm_text: '' } }),
      });
    });
    await page.route('**/v1/query', async (route: any) => {
      await route.fulfill({
        status: 200, contentType: 'application/json',
        body: JSON.stringify({ answer: 'Trả lời voice rút gọn.', contexts: [] }),
      });
    });
    await page.route('**/v1/glucose/followup', async (route: any) => {
      await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ saved: true }) });
    });
    await page.route('**/v1/voice/speak', async (route: any) => {
      await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ audio_available: false, text: '' }) });
    });

    await page.goto('/tracker');
    await page.locator('input[type="number"]').first().fill('140');
    await page.getByRole('button', { name: /Lưu chỉ số/ }).click();
    await expect(page.getByText(/Hỏi thêm ngữ cảnh/)).toBeVisible({ timeout: 5000 });
    await expect(page.getByTestId('mic-btn')).toBeVisible();
    await page.getByTestId('mic-btn').click();
    await page.getByTestId('mic-btn').click();
    await expect(page.getByText('Trả lời voice rút gọn.')).toBeVisible({ timeout: 10000 });
    await expect(page.getByPlaceholder('vd: ăn 2 miếng bánh ngọt')).toBeVisible();
  });

  test('voice STT failure shows sys message and no auto-send', async ({ page }) => {
    await page.addInitScript(() => {
      const g = navigator as any;
      if (!g.mediaDevices) (g as any).mediaDevices = {};
      (g.mediaDevices as any).getUserMedia = async () => ({ getTracks: () => [] });
      (window as any).MediaRecorder = class {
        ondataavailable: any = null;
        onstop: any = null;
        stream: any;
        constructor(stream: any) { this.stream = stream; }
        start() {}
        stop() { if (this.onstop) this.onstop(); }
      };
    });
    await mockSession(page);
    await mockGlucosePost(page, {
      anomaly: { type: 'trend', direction: 'up', reason: 'cascade test' },
      follow_up_questions: ['Khẩu phần ăn gần đây thế nào?'],
    });
    await mockGlucoseGet(page, []);
    await page.route('**/v1/voice/transcribe', async (route: any) => {
      await route.fulfill({ status: 500, contentType: 'application/json', body: JSON.stringify({ detail: 'transcribe failed' }) });
    });
    let queryCalls = 0;
    await page.route('**/v1/query', async (route: any) => {
      queryCalls += 1;
      await route.fulfill({
        status: 200, contentType: 'application/json',
        body: JSON.stringify({ answer: 'Không được gọi.', contexts: [] }),
      });
    });
    await page.route('**/v1/voice/speak', async (route: any) => {
      await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ audio_available: false, text: '' }) });
    });

    await page.goto('/tracker');
    await page.locator('input[type="number"]').first().fill('140');
    await page.getByRole('button', { name: /Lưu chỉ số/ }).click();
    await expect(page.getByText(/Hỏi thêm ngữ cảnh/)).toBeVisible({ timeout: 5000 });
    await expect(page.getByTestId('mic-btn')).toBeVisible();
    await page.getByTestId('mic-btn').click();
    await page.getByTestId('mic-btn').click();
    await expect(page.getByText(/Không nghe rõ/)).toBeVisible({ timeout: 10000 });
    await expect(page.getByText('Không được gọi.')).toBeHidden();
    expect(queryCalls).toBe(0);
  });
});
