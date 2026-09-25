import { test, expect } from './fixtures';

// Pre-visit B — PrevisitView (auth via useAuth, patientId = session user).
// Session mocked through /v1/auth/me; BE not started by webServer so API is mocked.
const ME = { id: 'soap_user', username: 'soap_user', role: 'user', is_active: true, is_verified: true };

function mockSession(page: any) {
  return page.route('**/v1/auth/me', async (route: any) => {
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(ME) });
  });
}

function mockGlucoseGet(page: any) {
  return page.route('**/v1/glucose/*', async (route: any) => {
    await route.fulfill({
      status: 200, contentType: 'application/json',
      body: JSON.stringify({
        user_id: ME.id,
        logs: [{ id: 1, user_id: ME.id, value_mgdl: 130, measured_at: '2026-09-06T10:00:00', context: 'fasting', notes: 'banh ngot', classification: 'high' }],
        stats: { user_id: ME.id, total_logs: 1, avg_mgdl: 130, last_7_days_avg: 130, streak_days: 1, logs_per_week: 1, classification_counts: { high: 1 } },
        should_escalate: false,
        anomaly_ids: [1],
      }),
    });
  });
}

function mockSoapJson(page: any, soap: any) {
  return page.route('**/v1/soap/generate', async (route: any) => {
    if (route.request().url().includes('/markdown')) {
      await route.continue();
      return;
    }
    await route.fulfill({
      status: 200, contentType: 'application/json',
      body: JSON.stringify({
        user_id: ME.id,
        generated_at: new Date().toISOString(),
        period: '14 ngày',
        soap,
        stats: { user_id: ME.id, total_logs: 1, avg_mgdl: 130, last_7_days_avg: 130, streak_days: 1, logs_per_week: 1, classification_counts: { high: 1 } },
      }),
    });
  });
}

test.describe('Pre-visit B — SOAP Generation', () => {
  test('shows empty state when no SOAP yet', async ({ page }) => {
    await mockSession(page);
    await page.goto('/previsit');
    await expect(page.getByText(/Nhập logs ở Tracker trước rồi tạo SOAP/)).toBeVisible();
  });

  test('generate SOAP JSON shows sections and empty-plan placeholder', async ({ page }) => {
    await mockSession(page);
    await mockGlucoseGet(page);
    await mockSoapJson(page, {
      subjective: 'Người bệnh ghi nhận ăn ngọt [Xem log #1]',
      objective: 'Trung bình 130 mg/dL [Xem log #1]',
      assessment: 'Chưa đạt mục tiêu HbA1c<7% [Xem log #1]',
      plan: '',
    });

    await page.goto('/previsit');
    await expect(page.getByTestId('soap-json-btn')).toBeVisible();
    await page.getByRole('button', { name: /Tạo SOAP \(JSON\)/ }).click();
    await expect(page.getByText('S — Subjective')).toBeVisible({ timeout: 5000 });
    await expect(page.getByText('O — Objective')).toBeVisible();
    await expect(page.getByText('A — Assessment')).toBeVisible();
    await expect(page.getByText('Dành cho bác sĩ chỉ định')).toBeVisible();
    await expect(page.getByTestId('log-link').first()).toBeVisible();
    await expect(page.locator('a[href*="/tracker?highlight="]').first()).toBeVisible();
  });

  test('generate markdown and download', async ({ page }) => {
    await mockSession(page);
    await mockGlucoseGet(page);
    const md = '# SOAP — soap_user\n\n## S\nTest';
    await page.route('**/v1/soap/generate/markdown', async (route: any) => {
      await route.fulfill({ status: 200, contentType: 'text/plain', body: md });
    });

    await page.goto('/previsit');
    await page.getByRole('button', { name: /Tạo Markdown/ }).click();
    const [download] = await Promise.all([
      page.waitForEvent('download'),
      page.getByRole('button', { name: /Tải \.md/ }).click(),
    ]);
    expect(download.suggestedFilename()).toMatch(/SOAP_/);
  });

  test('handles API error gracefully', async ({ page }) => {
    await mockSession(page);
    await mockGlucoseGet(page);
    await page.route('**/v1/soap/generate', async (route: any) => {
      if (route.request().url().includes('/markdown')) {
        await route.continue();
        return;
      }
      await route.fulfill({ status: 500, contentType: 'application/json', body: JSON.stringify({ detail: 'Internal' }) });
    });

    await page.goto('/previsit');
    await page.getByRole('button', { name: /Tạo SOAP \(JSON\)/ }).click();
    await expect(page.getByText(/soap failed 500/)).toBeVisible({ timeout: 5000 });
  });
});
