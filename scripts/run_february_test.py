import os
import sys
import pandas as pd
import numpy as np

# Force UTF-8 on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure root directory is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.agent.pipeline import SamrukWindAgentPipeline

def run_test():
    print("=" * 70)
    print("[TEST RUN] HackAlem AI: Запуск полного тестового прогона за Февраль 2026 г.")
    print("Объект: ВЭС Шелек (Турбина 1 [2.5 МВт], Турбина 2 [2.5 МВт], ВЭС [5.0 МВт])")
    print("=" * 70)

    pipe = SamrukWindAgentPipeline()

    all_rows = []
    daily_summaries = []

    # Iterate over all 28 days of February 2026
    for day in range(1, 29):
        date_str = f"2026-02-{day:02d}"
        
        # Test 1: Single Turbine 1 (24h)
        res_t1 = pipe.run_forecast_cycle(target_date=date_str, horizon_hours=24, turbine_id="turbine_1")
        # Test 2: Single Turbine 2 (24h)
        res_t2 = pipe.run_forecast_cycle(target_date=date_str, horizon_hours=24, turbine_id="turbine_2")
        # Test 3: Total Farm (48h horizon)
        res_farm_48 = pipe.run_forecast_cycle(target_date=date_str, horizon_hours=48, turbine_id="farm")

        audit = res_farm_48["audit"]
        bench = res_farm_48["benchmark"]

        daily_summaries.append({
            "date": date_str,
            "t1_gen_24h_mwh": res_t1["audit"]["total_generation_mwh"],
            "t2_gen_24h_mwh": res_t2["audit"]["total_generation_mwh"],
            "farm_gen_24h_mwh": round(res_t1["audit"]["total_generation_mwh"] + res_t2["audit"]["total_generation_mwh"], 2),
            "farm_gen_48h_mwh": audit["total_generation_mwh"],
            "capacity_factor_pct": audit["capacity_factor_pct"],
            "saved_kegoc_penalties_kzt": audit["estimated_penalty_saved_kzt"],
            "skill_score": bench["skill_score_index"],
            "agent_status": audit["agent_status"],
            "events_count": len(audit.get("events", []))
        })

        for row in res_farm_48["timeline"]:
            all_rows.append({
                "forecast_origin_date": date_str,
                "lead_hour": row["hour_index"] + 1,
                "timestamp": row["timestamp"],
                "turbine_id": "farm",
                "predicted_mw": row["predicted_mwh"],
                "physics_baseline_mw": row["physics_mwh"],
                "persistence_baseline_mw": row["persistence_mwh"],
                "wind_speed_100m_ms": row["wind_speed_100m"],
                "temperature_2m_c": row["temperature_2m"],
                "air_density_kg_m3": row["air_density_kg_m3"]
            })

    # Save full hourly predictions to CSV
    out_csv = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "submission_forecast_february_2026.csv")
    df_pred = pd.DataFrame(all_rows)
    df_pred.to_csv(out_csv, index=False)

    df_sum = pd.DataFrame(daily_summaries)
    
    print("\n[SUCCESS] Тестовый прогон 28 дней февраля 2026 года успешно завершен!")
    print(f"[OUTPUT] Итоговый файл прогнозов: {out_csv} (Строк: {len(df_pred):,})")
    print("\n--- СВОДНЫЕ РЕЗУЛЬТАТЫ ЗА ФЕВРАЛЬ 2026 ---")
    print(f"Всего суток: {len(df_sum)}")
    print(f"Суммарная выработка ВЭС (48ч сумма): {df_sum['farm_gen_48h_mwh'].sum():,.2f} МВт·ч")
    print(f"Средний суточный КУИМ: {df_sum['capacity_factor_pct'].mean():.1f}%")
    print(f"Предотвращенные штрафы KEGOC: {df_sum['saved_kegoc_penalties_kzt'].sum():,} ₸")
    print(f"Средний индекс Skill Score (vs Persistence): {df_sum['skill_score'].mean():.3f}")
    
    critical_days = df_sum[df_sum["agent_status"] == "CRITICAL_SHUTDOWN"]
    print(f"Дней со штормовым остановом: {len(critical_days)}")
    advisory_days = df_sum[df_sum["agent_status"] == "ADVISORY_ATTENTION"]
    print(f"Дней с обледенением/градиентами ветра: {len(advisory_days)}")
    print("=" * 70)

if __name__ == "__main__":
    run_test()
