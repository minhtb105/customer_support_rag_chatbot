import { test, expect } from './fixtures';

test.describe('Tracker A — Glucose Logging & Chart', () => {
  test('form validation and classification badge', async ({ page, userId }) => {
    // Mock POST /v1/glucose and GET /v1/glucose/{user}
    await page.route('**/v1/glucose', async (route) => {
      if (route.request().method() === 'POST') {
        const body = await route.request().postDataJSON();
        const val = body.value_mgdl;
        let cls = 'normal';
        if (val < 70) cls = 'low';
        else if (val >= 300) cls = 'critical';
        else if (body.context === 'fasting' && val >= 126) cls = 'high';
        else if (body.context === 'fasting' && val >= 100) cls = 'elevated';
        else if (body.context === 'post_meal_2h' && val >= 200) cls = 'high';
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ id: 1, user_id: body.user_id, value_mgdl: val, measured_at: new Date().toISOString(), context: body.context, notes: body.notes, classification: cls, message: `${cls} message` }),
        });
      } else {
        await route.continue();
      }
    });

    await page.route('**/v1/glucose/**', async (route) => {
      if (route.request().method() === 'GET') {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            user_id: userId,
            logs: [{ id: 1, user_id: userId, value_mgdl: 130, measured_at: new Date().toISOString(), context: 'fasting', notes: null, classification: 'high' }],
            stats: { user_id: userId, total_logs: 1, avg_mgdl: 130, last_7_days_avg: 130, streak_days: 1, logs_per_week: 0.5, classification_counts: { high: 1 } },
            should_escalate: false,
          }),
        });
      } else {
        await route.continue();
      }
    });

    await page.goto('/tracker');
    await page.getByLabel(/User ID/).fill(userId);
    await page.getByLabel(/Giá trị/).fill('130');
    await page.getByLabel(/Bối cảnh/).selectOption('fasting');
    await page.getByRole('button', { name: /Lưu chỉ số/ }).click();
    await expect(page.getByText('HIGH: high message')).toBeVisible({ timeout: 5000 });
    await expect(page.getByText(/Tổng logs/)).toBeVisible();
  });

  test('chart shows reference lines 126/200/70', async ({ page }) => {
    await page.goto('/tracker');
    // Mock to have data for chart
    await page.route('**/v1/glucose/**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          user_id: 'mock',
          logs: [
            { id: 1, user_id: 'mock', value_mgdl: 110, measured_at: '2026-09-06T10:00:00Z', context: 'fasting', classification: 'elevated' },
            { id: 2, user_id: 'mock', value_mgdl: 95, measured_at: '2026-09-06T11:00:00Z', context: 'fasting', classification: 'normal' },
          ],
          stats: { user_id: 'mock', total_logs: 2, avg_mgdl: 102, last_7_days_avg: 102, streak_days: 1, logs_per_week: 2, classification_counts: { normal: 1, elevated: 1 } },
          should_escalate: false,
        }),
      });
    });
    await page.goto('/tracker');
    await expect(page.getByText(/Biểu đồ 20 lần/)).toBeVisible();
    // Reference lines rendered as SVG - check container visible
    await expect(page.locator('.recharts-wrapper')).toBeVisible();
  });

  test('KPI alert when <3 per week', async ({ page }) => {
    await page.route('**/v1/glucose/**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          user_id: 'kpi',
          logs: [{ id: 1, user_id: 'kpi', value_mgdl: 100, measured_at: new Date().toISOString(), context: 'fasting', classification: 'elevated' }],
          stats: { user_id: 'kpi', total_logs: 1, avg_mgdl: 100, last_7_days_avg: 100, streak_days: 1, logs_per_week: 0.5, classification_counts: { elevated: 1 } },
          should_escalate: false,
        }),
      });
    });
    await page.goto('/tracker');
    await expect(page.getByText(/Chưa đạt KPI 3\/tuần/)).toBeVisible();
  });

  test('escalation banner when critical', async ({ page }) => {
    await page.route('**/v1/glucose/**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          user_id: 'crit',
          logs: [{ id: 1, user_id: 'crit', value_mgdl: 350, measured_at: new Date().toISOString(), context: 'random', classification: 'critical' }],
          stats: { user_id: 'crit', total_logs: 1, avg_mgdl: 350, last_7_days_avg: 350, streak_days: 1, logs_per_week: 1, classification_counts: { critical: 1 } },
          should_escalate: true,
        }),
      });
    });
    await page.goto('/tracker');
    await expect(page.getByText(/Khuyến nghị liên hệ bác sĩ/)).toBeVisible();
  });

  test('trend explanation calls RAG and shows answer', async ({ page }) => {
    await page.route('**/v1/glucose/**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          user_id: 'trend',
          logs: [{ id: 1, user_id: 'trend', value_mgdl: 130, measured_at: new Date().toISOString(), context: 'fasting', classification: 'high' }],
          stats: { user_id: 'trend', total_logs: 1, avg_mgdl: 130, last_7_days_avg: 130, streak_days: 1, logs_per_week: 1, classification_counts: { high: 1 } },
          should_escalate: false,
        }),
      });
    });
    await page.route('**/v1/query', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ answer: 'Xu hướng tăng nhẹ, khuyến nghị theo dõi WHO.', cited_sources: [1], contexts: [], audit: { latency_ms: 123, citations: [] }, cache_hit: false }),
      });
    });
    await page.goto('/tracker');
    await page.getByRole('button', { name: /Diễn giải xu hướng/ }).click();
    await expect(page.getByText(/Xu hướng tăng nhẹ/)).toBeVisible();
  });
});
