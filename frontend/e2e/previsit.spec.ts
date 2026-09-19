import { test, expect } from './fixtures';

test.describe('Pre-visit B — SOAP Generation', () => {
  test('generate SOAP JSON shows 4 sections', async ({ page }) => {
    await page.route('**/v1/soap/generate', async (route) => {
      // Handle markdown endpoint separately
      if (route.request().url().includes('/markdown')) {
        await route.continue();
        return;
      }
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          user_id: 'soap_user',
          generated_at: new Date().toISOString(),
          period: '14 ngày',
          soap: {
            subjective: 'Người bệnh ghi nhận 5 lần đo...',
            objective: 'Chỉ số trung bình 130 mg/dL...',
            assessment: 'Chưa đạt KPI...',
            plan: 'Duy trì đo ≥3 lần/tuần...',
          },
          stats: { user_id: 'soap_user', total_logs: 5, avg_mgdl: 130, last_7_days_avg: 128, streak_days: 3, logs_per_week: 2.5, classification_counts: { high: 2 } },
        }),
      });
    });

    await page.goto('/previsit');
    await page.getByLabel(/User ID/).fill('soap_user');
    await page.getByRole('button', { name: /Tạo SOAP \(JSON\)/ }).click();
    await expect(page.getByText('S — Subjective')).toBeVisible();
    await expect(page.getByText('O — Objective')).toBeVisible();
    await expect(page.getByText('A — Assessment')).toBeVisible();
    await expect(page.getByText('P — Plan')).toBeVisible();
    await expect(page.getByText(/Tổng logs/)).toBeVisible();
  });

  test('generate markdown and download', async ({ page }) => {
    const md = '# Tóm tắt trước tái khám — soap_user\n\n## S — Subjective\nTest';
    await page.route('**/v1/soap/generate/markdown', async (route) => {
      await route.fulfill({ status: 200, contentType: 'text/plain', body: md });
    });
    await page.route('**/v1/soap/generate', async (route) => {
      if (route.request().url().includes('/markdown')) {
        await route.continue();
        return;
      }
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          user_id: 'soap_user',
          generated_at: new Date().toISOString(),
          period: '7 ngày',
          soap: { subjective: 's', objective: 'o', assessment: 'a', plan: 'p' },
          stats: { user_id: 'soap_user', total_logs: 1, avg_mgdl: 100, last_7_days_avg: 100, streak_days: 1, logs_per_week: 1, classification_counts: {} },
        }),
      });
    });

    await page.goto('/previsit');
    await page.getByLabel(/User ID/).fill('soap_user');
    await page.getByRole('button', { name: /Tạo Markdown/ }).click();
    await expect(page.getByText(/Markdown — sẵn sàng/)).toBeVisible();
    const [download] = await Promise.all([
      page.waitForEvent('download'),
      page.getByRole('button', { name: /Tải \.md/ }).click(),
    ]);
    expect(download.suggestedFilename()).toMatch(/SOAP_/);
  });

  test('shows empty state when no logs', async ({ page }) => {
    await page.goto('/previsit');
    await expect(page.getByText(/Nhập User ID đã có logs/)).toBeVisible();
  });

  test('handles API error gracefully', async ({ page }) => {
    await page.route('**/v1/soap/generate', async (route) => {
      await route.fulfill({ status: 500, contentType: 'application/json', body: JSON.stringify({ detail: 'Internal' }) });
    });
    await page.goto('/previsit');
    await page.getByRole('button', { name: /Tạo SOAP \(JSON\)/ }).click();
    await expect(page.getByText(/Internal/)).toBeVisible();
  });
});
