import { test as base, expect } from '@playwright/test';

// No shared fixtures: specs import { test, expect } from here for a single
// import path. (userId fixture removed — 0 consumers.)

export const test = base;

export { expect };
