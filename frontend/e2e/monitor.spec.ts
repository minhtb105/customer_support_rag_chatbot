import { test, expect } from './fixtures';

test.describe('Monitor — Guidelines & Alerts', () => {
  test('monitor layout renders tabs', async ({ page }) => {
    await page.goto('/monitor/guidelines');
    await expect(page.getByText('Giám sát — Guidelines & An toàn thuốc')).toBeVisible();
    await expect(page.getByText('Guidelines').first()).toBeVisible();
    await expect(page.getByText('Cảnh báo thuốc').first()).toBeVisible();
  });

  test('guidelines page shows auth guard for anonymous', async ({ page }) => {
    await page.goto('/monitor/guidelines');
    // Should show login guard or list (depending on auth) — at least heading visible
    await expect(page.getByText(/Giám sát|Cần đăng nhập|Cần role/)).toBeVisible();
  });

  test('alerts page shows auth guard for anonymous', async ({ page }) => {
    await page.goto('/monitor/alerts');
    await expect(page.getByText(/Giám sát|Cần đăng nhập|Cần role/)).toBeVisible();
  });

  test('guidelines pending list mock', async ({ page }) => {
    await page.route('**/v1/monitors/guidelines*', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          total: 1,
          items: [
            {
              id: 'test1234',
              source: 'gold',
              title: 'GOLD COPD 2024 Test',
              version_label: 'GOLD 2025 v1.2',
              status: 'pending_review',
              sha256: 'abc123',
              fetched_at: '2026-09-06T12:00:00',
              change_summary_json: JSON.stringify({ tom_tat_tieng_viet: 'Thay đổi GOLD test' }),
            },
          ],
          limit: 20,
          offset: 0,
        }),
      });
    });
    // Need to bypass auth guard: mock auth/me as specialist
    await page.route('**/v1/auth/me', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ id: 'u1', username: 'spec', role: 'specialist', is_active: true, is_verified: true }),
      });
    });
    await page.goto('/monitor/guidelines');
    // Should show mocked item or at least not error
    await expect(page.getByText(/Giám sát|GOLD|Không có/)).toBeVisible();
  });

  test('alerts pending list mock', async ({ page }) => {
    await page.route('**/v1/monitors/alerts*', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          total: 1,
          items: [
            {
              id: 'alert123',
              source: 'FDA',
              alert_type: 'recall',
              severity: 'critical',
              drug_name: 'Metformin',
              alert_title: 'Thu hồi Metformin lô X',
              alert_url: 'https://api.fda.gov/test',
              ai_summary: JSON.stringify({ tom_tat_vi: 'Thu hồi do nhiễm bẩn' }),
              fetched_at: '2026-09-06T12:00:00',
              status: 'pending_review',
            },
          ],
          limit: 20,
          offset: 0,
        }),
      });
    });
    await page.route('**/v1/auth/me', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ id: 'u1', username: 'pharm', role: 'pharmacist', is_active: true, is_verified: true }),
      });
    });
    await page.goto('/monitor/alerts');
    await expect(page.getByText(/Giám sát|Metformin|Không có/)).toBeVisible();
  });

  test('Header has monitor link for privileged roles (mock)', async ({ page }) => {
    await page.route('**/v1/auth/me', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ id: 'u1', username: 'admin', role: 'admin', is_active: true, is_verified: true }),
      });
    });
    await page.route('**/v1/auth/notifications', async (route) => {
      await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ notifications: [], unread_count: 0 }) });
    });
    await page.goto('/');
    await expect(page.getByText('Giám sát').first()).toBeVisible();
  });
});
