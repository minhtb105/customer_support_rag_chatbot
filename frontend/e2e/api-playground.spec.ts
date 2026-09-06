import { test, expect } from './fixtures';

test.describe('API Playground C — WHO-RAG Query', () => {
  test('submits diabetes query and shows audit trail', async ({ page }) => {
    await page.route('**/v1/query', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          answer: 'Ngưỡng chẩn đoán đái tháo đường: đói ≥126 mg/dL [Source 1].',
          cited_sources: [1],
          contexts: [],
          audit: {
            cited_sources: [1],
            citations: [{ source_id: '1', dataset: 'WHO_Classification_Diabetes_2019.pdf', content_snippet: 'fasting >=126', score: 0.9 }],
            prompt_version: 'abc123',
            reranker_model: 'cross-encoder/ms-marco-MiniLM-L-6-v2',
            latency_ms: 234,
          },
          cache_hit: false,
        }),
      });
    });

    await page.goto('/api-playground');
    await expect(page.getByText(/WHO-RAG Infrastructure API/)).toBeVisible();
    await page.getByRole('button', { name: /Gửi truy vấn/ }).click();
    await expect(page.getByText(/Ngưỡng chẩn đoán/)).toBeVisible();
    await expect(page.getByText(/Cited: \[1\]/)).toBeVisible();
    await expect(page.getByText(/Audit Trail/)).toBeVisible();
    await expect(page.getByText(/abc123/)).toBeVisible();
  });

  test('example buttons fill query', async ({ page }) => {
    await page.goto('/api-playground');
    await page.getByRole('button', { name: /Ngưỡng chẩn đoán đái tháo đường theo WHO/ }).click();
    await expect(page.locator('textarea')).toHaveValue(/Ngưỡng chẩn đoán/);
  });

  test('top_k slider changes value', async ({ page }) => {
    await page.goto('/api-playground');
    const slider = page.locator('input[type="range"]');
    await expect(slider).toHaveValue('5');
    await slider.fill('8');
    await expect(page.getByText(/Top K: 8/)).toBeVisible();
  });

  test('shows raw JSON for B2B integration', async ({ page }) => {
    await page.route('**/v1/query', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ answer: 'Test', cited_sources: [], contexts: [], audit: { citations: [] } }),
      });
    });
    await page.goto('/api-playground');
    await page.getByRole('button', { name: /Gửi truy vấn/ }).click();
    await expect(page.getByText(/Raw JSON/)).toBeVisible();
    await expect(page.getByText(/curl -X POST/)).toBeVisible();
  });

  test('handles RAG error', async ({ page }) => {
    await page.route('**/v1/query', async (route) => {
      await route.fulfill({ status: 500, contentType: 'application/json', body: JSON.stringify({ detail: 'RAG error' }) });
    });
    await page.goto('/api-playground');
    await page.getByRole('button', { name: /Gửi truy vấn/ }).click();
    await expect(page.getByText(/RAG error/)).toBeVisible();
  });

  test('cache hit badge', async ({ page }) => {
    await page.route('**/v1/query', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ answer: 'Cached', cited_sources: [], contexts: [], cache_hit: true, audit: { citations: [] } }),
      });
    });
    await page.goto('/api-playground');
    await page.getByRole('button', { name: /Gửi truy vấn/ }).click();
    await expect(page.getByText(/cache hit/)).toBeVisible();
  });
});
