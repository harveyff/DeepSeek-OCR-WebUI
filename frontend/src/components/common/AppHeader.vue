<template>
  <NLayoutHeader
    data-testid="app-header"
    class="app-header"
    bordered
  >
    <!-- Left: Branding -->
    <div class="header-brand">
      <NIcon
        size="24"
        :color="PRIMARY_COLOR"
        class="brand-icon"
      >
        <DocumentText />
      </NIcon>
      <span
        class="app-title"
        data-testid="app-title"
      >
        {{ $t('header.scan2Doc') }}
      </span>
    </div>

    <!-- Center: OCR Status & Queue -->
    <div class="header-center">
      <template v-if="store.ocrTaskCount === 0 && !showQueue">
        <OCRHealthIndicator />
      </template>

      <NPopover
        v-if="store.ocrTaskCount > 0 || showQueue"
        v-model:show="showQueue"
        trigger="manual"
        placement="bottom"
        :show-arrow="false"
        raw
      >
        <template #trigger>
          <div
            v-show="store.ocrTaskCount > 0"
            class="status-pill"
            data-testid="ocr-queue-badge"
            @click.stop="showQueue = !showQueue"
          >
            <NSpin
              size="small"
              :stroke-width="20"
              class="status-spinner"
            />
            <span class="status-text">
              {{ $t('header.processing') }}: {{ store.activeOCRTasks.length }} | {{ $t('header.waiting') }}: {{ store.queuedOCRTasks.length }}
            </span>
            <NDivider vertical />
            <OCRHealthIndicator compact />
          </div>
        </template>
        <div ref="popoverContentRef">
          <OCRQueuePopover @close="showQueue = false" />
        </div>
      </NPopover>
    </div>

    <!-- Right: Actions -->
    <div class="header-actions">
      <!-- Language Selector -->
      <LanguageSelector />

      <!-- Primary CTA -->
      <NButton
        type="primary"
        size="medium"
        class="add-btn"
        data-testid="import-files-button"
        @click="handleAddFiles"
      >
        <template #icon>
          <NIcon>
            <CloudUpload />
          </NIcon>
        </template>
        {{ $t('header.importFiles') }}
      </NButton>
    </div>
  </NLayoutHeader>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { onClickOutside } from '@vueuse/core'
import { NLayoutHeader, NButton, NIcon, NPopover, NSpin, NDivider, NTooltip } from 'naive-ui'
import { DocumentText, CloudUpload } from '@vicons/ionicons5'
import { usePagesStore } from '@/stores/pages'
import OCRQueuePopover from '@/components/common/OCRQueuePopover.vue'
import OCRHealthIndicator from '@/components/common/OCRHealthIndicator.vue'
import LanguageSelector from '@/components/common/LanguageSelector.vue'
import { PRIMARY_COLOR } from '@/theme/vars'

// eslint-disable-next-line @typescript-eslint/no-empty-object-type
defineProps<{}>()

const emit = defineEmits<{
  (e: 'add-files'): void
}>()

useI18n()
const store = usePagesStore()
const showQueue = ref(false)
const popoverContentRef = ref<HTMLElement | null>(null)

// Intelligent Dismissal: Close queue when clicking outside, but ignore:
// 1. The trigger badge itself (handled by its own click)
// 2. Elements with .keep-queue-open class (Scan buttons, Quick Actions)
onClickOutside(popoverContentRef, () => {
  showQueue.value = false
}, {
  ignore: [
    '.keep-queue-open', 
    '[data-testid="ocr-queue-badge"]',
    '[data-testid="ocr-trigger-btn"]',
    '[data-testid="ocr-mode-dropdown"]',
    '[data-testid="ocr-actions-container"]',
    '.ocr-actions',
    '.ocr-mode-selector'
  ]
})

const handleAddFiles = () => {
  emit('add-files')
}

defineExpose({
  showQueue,
  handleAddFiles
})
</script>

<style scoped>
.app-header {
  height: 64px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 24px;
  background-color: #f6f7f8;
  position: relative; /* For absolute center positioning */
}

.header-brand {
  display: flex;
  align-items: center;
  gap: 12px;
  user-select: none;
  z-index: 1;
}

.brand-icon {
  display: flex;
}

.app-title {
  font-size: 18px;
  font-weight: 700;
  color: #333;
  letter-spacing: -0.5px;
}

.header-center {
  position: absolute;
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  align-items: center;
  z-index: 2;
}

.status-pill {
  display: flex;
  align-items: center;
  gap: 8px;
  background: white;
  border: 1px solid #e0e0e0;
  border-radius: 20px;
  padding: 6px 16px;
  cursor: pointer;
  font-size: 13px;
  font-weight: 500;
  color: #333;
  box-shadow: 0 1px 3px rgba(0,0,0,0.05);
  transition: all 0.2s;
}

.status-pill:hover {
  background: #fafafa;
  box-shadow: 0 2px 6px rgba(0,0,0,0.08);
}

.status-spinner {
  --n-size: 14px !important;
}

.header-actions {
  display: flex;
  align-items: center;
  gap: 16px;
  z-index: 1;
}

@media (max-width: 768px) {
  .app-header {
    padding: 0 12px;
    height: 56px;
  }

  .app-title {
    font-size: 14px;
    white-space: nowrap;
  }

  .header-center {
    display: none;
  }

  .header-actions {
    gap: 8px;
  }

  .header-actions .n-divider {
    display: none;
  }

  .add-btn {
    padding: 0 12px;
    font-size: 13px;
  }

  .add-btn .n-button__content {
    gap: 4px;
  }
}

@media (max-width: 480px) {
  .app-header {
    padding: 0 8px;
  }

  .header-brand {
    gap: 8px;
  }

  .app-title {
    font-size: 13px;
  }

  .add-btn span {
    display: none;
  }
}
</style>
