import { test, expect } from './fixtures';

test.describe('Overview — Market Evidence & Navigation', () => {
  test('renders hero stats and 5-criteria framework', async ({ page }) => {
    await page.goto('/');
    await expect(page.getByText('Diabetes RAG')).toBeVisible();
    await expect(page.getByText('7,3%').first()).toBeVisible();
    await expect(page.getByText('20,2%').first()).toBeVisible();
    await expect(page.getByText('14/10k').first()).toBeVisible();
    // 5 criteria grid
    await expect(page.getByText(/Pain Point/)).toBeVisible();
  });

  test('shows diabetes adherence table and 3 track cards', async ({ page }) => {
    await page.goto('/');
    await expect(page.getByText('BV Thanh Nhàn 2025')).toBeVisible();
    await expect(page.getByText('Trợ lý tuân thủ')).toBeVisible();
    await expect(page.getByText('Hồ sơ trước tái khám')).toBeVisible();
    await expect(page.getByText('WHO-RAG API').first()).toBeVisible();
  });

  test('navigation to 3 tracks works', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    await expect(page.locator('a[href*="/tracker"]').first()).toBeVisible();
    await expect(page.locator('a[href*="/previsit"]').first()).toBeVisible();
    await expect(page.locator('a[href*="/api-playground"]').first()).toBeVisible();
    // Direct navigation verifies routes exist and render
    await page.goto('/tracker');
    await expect(page.getByText(/Nhật ký đường huyết/)).toBeVisible();
    await page.goto('/previsit');
    await expect(page.getByText(/Hồ sơ trước tái khám/)).toBeVisible();
    await page.goto('/api-playground');
    await expect(page.getByText(/WHO-RAG Infrastructure API/)).toBeVisible();
  });

  test('displays corpus status or fallback when API offline', async ({ page }) => {
    // Mock guidelines status to ensure deterministic
    await page.route('**/v1/guidelines/status', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ total_pdfs: 22, by_source: { _root: 18, diabetes: 4 }, manifest_path: 'data/guideline_manifest.json' }),
      });
    });
    await page.goto('/');
    await expect(page.getByText(/Tổng 22 PDFs|Corpus Guideline/)).toBeVisible();
  });
});
