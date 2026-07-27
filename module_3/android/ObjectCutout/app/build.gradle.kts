plugins {
    id("com.android.application")
}

android {
    namespace = "org.arm.education.executorch.objectcutout"
    compileSdk = 35

    defaultConfig {
        applicationId = "org.arm.education.executorch.objectcutout"
        minSdk = 28
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"

        ndk {
            abiFilters += listOf("arm64-v8a", "x86_64")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

dependencies {
    val executorchVersion = "1.3.1"
    implementation("org.pytorch:executorch-android:$executorchVersion")
}
