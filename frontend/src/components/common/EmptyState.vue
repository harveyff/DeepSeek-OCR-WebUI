<template>
  <div class="empty-state-container">
    <div class="empty-state-hero">
      <div class="empty-icon-container">
        <NIcon
          size="64"
          :color="PRIMARY_COLOR"
        >
          <CloudUploadOutline />
        </NIcon>
      </div>
      <h2
        class="empty-title"
        data-testid="welcome-title"
      >
        {{ $t('app.welcomeTitle') }}
      </h2>
      <p
        class="empty-description"
        data-testid="welcome-description"
      >
        {{ $t('app.welcomeDescription') }}
      </p>
      <NButton
        type="primary"
        size="large"
        class="select-files-btn empty-add-btn"
        data-testid="start-import-button"
        @click="$emit('add-files')"
      >
        <template #icon>
          <NIcon>
            <AddOutline />
          </NIcon>
        </template>
        {{ $t('app.startImport') }}
      </NButton>
    </div>
  </div>
</template>

<script setup lang="ts">
import { NIcon, NButton } from 'naive-ui'
import { CloudUploadOutline, AddOutline } from '@vicons/ionicons5'
import { PRIMARY_COLOR } from '@/theme/vars'

defineEmits<{
  (e: 'add-files'): void
}>()
</script>

<style scoped>
.empty-state-container {
  height: 100%;
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: #f9fafb;
}

.empty-state-hero {
  text-align: center;
  padding: 40px;
  max-width: 600px;
}

.icon-wrapper {
  margin-bottom: 24px;
  animation: float 6s ease-in-out infinite;
}

.hero-title {
  font-size: 32px;
  font-weight: 800;
  color: #333;
  margin: 0 0 16px 0;
  letter-spacing: -1px;
}

.hero-subtitle {
  font-size: 18px;
  color: #666;
  margin: 0 0 32px 0;
}

.select-files-btn {
  font-weight: 600;
  padding: 0 32px;
  height: 48px;
  font-size: 16px;
  box-shadow: 0 4px 14px 0 rgba(24, 160, 88, 0.39);
}

@keyframes float {
  0% {
    transform: translateY(0px);
  }
  50% {
    transform: translateY(-10px);
  }
  100% {
    transform: translateY(0px);
  }
}

</style>
