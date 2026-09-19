import { test as base, expect } from '@playwright/test';

type Fixtures = {
  userId: string;
};

export const test = base.extend<Fixtures>({
  userId: async ({}, use) => {
    const id = `e2e_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`;
    await use(id);
  },
});

export { expect };
