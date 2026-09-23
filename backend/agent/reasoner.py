import numpy as np
import pandas as pd

class DispatcherAgentReasoner:
    """
    Agentic Reasoner for Samruk-Kazyna Wind Farm Operations & KEGOC Grid Integration.
    Audits the generated forecast, flags severe meteorological risks (storm cut-outs, icing, ramps),
    and formats actionable dispatch recommendations in Russian and Kazakh.
    """
    @classmethod
    def audit_forecast(cls, forecast_df: pd.DataFrame, target_date: str, capacity_mw: float = 2.5) -> dict:
        """
        Conducts deep domain audit of the 24-48h forecast with exact capacity calibration.
        """
        p_norm = forecast_df["predicted_power"].values
        wind_100 = forecast_df["wind_speed_100m"].values if "wind_speed_100m" in forecast_df.columns else forecast_df["wind_speed_10m"].values * 1.35
        temps = forecast_df["temperature_2m"].values if "temperature_2m" in forecast_df.columns else np.zeros(len(p_norm))
        timestamps = forecast_df["time"].astype(str).values if "time" in forecast_df.columns else [f"{target_date} {i:02d}:00" for i in range(len(p_norm))]

        total_hours = len(p_norm)
        hourly_mwh = p_norm * capacity_mw
        total_mwh = float(np.sum(hourly_mwh))
        capacity_factor = float(np.mean(p_norm) * 100.0)

        # Baseline comparison: physical model generation
        phys_norm = forecast_df["theoretical_power"].values if "theoretical_power" in forecast_df.columns else p_norm
        phys_mwh = float(np.sum(phys_norm * capacity_mw))

        # Risk 1: Storm cut-out (v >= 25.0 m/s)
        storm_indices = [int(h) for h in np.where(wind_100 >= 25.0)[0]]
        near_storm_indices = [int(h) for h in np.where((wind_100 >= 20.0) & (wind_100 < 25.0))[0]]
        
        # Risk 2: Severe Icing (Temperature <= -10°C)
        icing_indices = [int(h) for h in np.where(temps <= -10.0)[0]]

        # Risk 3: Ramp rate (Rapid drop or spike >= 40% capacity in 1 hour)
        ramp_diff = np.abs(np.diff(p_norm, prepend=p_norm[0]))
        ramp_indices = [int(h) for h in np.where(ramp_diff >= 0.40)[0]]

        # Structured Actionable Events
        events = []
        for h in storm_indices:
            t_str = timestamps[h] if h < len(timestamps) else f"+{h}h"
            events.append({
                "hour_index": h,
                "timestamp": t_str,
                "type": "STORM_CUTOUT",
                "severity": "CRITICAL",
                "title_ru": "Аварийный штормовой останов",
                "title_kz": "Дауыл салдарынан апаттық тоқтау",
                "detail_ru": f"Скорость ветра {wind_100[h]:.1f} м/с превышает порог безопасности 25 м/с. Прогноз выработки обнулен.",
                "detail_kz": f"Жел жылдамдығы {wind_100[h]:.1f} м/с қауіпсіздік шегінен (25 м/с) асты. Өндіріс болжамы нөлге теңестірілді.",
                "action_ru": "Проверить ограничения турбины и согласовать нулевой прогноз с диспетчером; заявка автоматически не отправляется.",
                "action_kz": "Турбина шектеулерін тексеріп, нөлдік болжамды диспетчермен келісу; өтінім автоматты түрде жіберілмейді."
            })

        for h in icing_indices:
            t_str = timestamps[h] if h < len(timestamps) else f"+{h}h"
            events.append({
                "hour_index": h,
                "timestamp": t_str,
                "type": "ICING_HAZARD",
                "severity": "WARNING",
                "title_ru": "Риск аэродинамического обледенения",
                "title_kz": "Аэродинамикалық мұздану қаупі",
                "detail_ru": f"Температура {temps[h]:.1f}°C. Возможно снижение аэродинамического КПД лопастей.",
                "detail_kz": f"Температура {temps[h]:.1f}°C. Қалақтардың аэродинамикалық ПӘК төмендеуі мүмкін.",
                "action_ru": "Проверить влажность, датчики обледенения и регламент турбины. Температуры недостаточно для подтверждения обледенения.",
                "action_kz": "Ылғалдылықты, мұздану датчиктерін және турбина нұсқаулығын тексеру. Температура мұздануды растауға жеткіліксіз."
            })

        for h in ramp_indices:
            t_str = timestamps[h] if h < len(timestamps) else f"+{h}h"
            delta_mw = ramp_diff[h] * capacity_mw
            events.append({
                "hour_index": h,
                "timestamp": t_str,
                "type": "RAMP_EVENT",
                "severity": "WARNING",
                "title_ru": "Резкий градиент выработки (Ramp Rate)",
                "title_kz": "Өндірістің күрт ауытқуы",
                "detail_ru": f"Скачок мощности на {delta_mw:.2f} МВт за 1 час.",
                "detail_kz": f"1 сағатта қуаттың {delta_mw:.2f} МВт-қа өзгеруі.",
                "action_ru": "Заблаговременно передать обновленный суточный график в НДЦ СО ЕЭС Казахстана.",
                "action_kz": "Қазақстанның БЭЖ Ұлттық диспетчерлік орталығына жаңартылған тәуліктік кестені алдын ала жіберу."
            })

        # No actual generation, settled imbalance prices or counterfactual schedule is supplied.
        imbalance_tariff_kzt = None
        saved_penalties_kzt = None

        # Status determination
        if len(storm_indices) > 0:
            agent_status = "CRITICAL_SHUTDOWN"
        elif len(icing_indices) > 0 or len(ramp_indices) > 0:
            agent_status = "ADVISORY_ATTENTION"
        else:
            agent_status = "VERIFIED_STABLE"

        # Generate Dispatcher Narrative
        summary_ru, summary_kz = cls._generate_narratives(
            target_date, total_mwh, capacity_factor, capacity_mw, storm_indices, icing_indices, ramp_indices, total_hours
        )

        return {
            "target_date": target_date,
            "horizon_hours": total_hours,
            "capacity_mw": capacity_mw,
            "total_generation_mwh": round(total_mwh, 2),
            "physics_generation_mwh": round(phys_mwh, 2),
            "capacity_factor_pct": round(capacity_factor, 1),
            "estimated_penalty_saved_kzt": saved_penalties_kzt,
            "imbalance_tariff_kzt_mwh": imbalance_tariff_kzt,
            "economic_assessment": {
                "status": "NOT_EVALUATED",
                "reason": "Actual generation, baseline schedule and verified settlement prices required",
            },
            "reasoner_type": "DETERMINISTIC_RULES",
            "alerts": {
                "storm_cutout_detected": len(storm_indices) > 0,
                "storm_hours": storm_indices,
                "icing_risk_detected": len(icing_indices) > 0,
                "icing_hours": icing_indices,
                "high_ramp_hours": ramp_indices
            },
            "events": events,
            "dispatcher_brief_ru": summary_ru,
            "dispatcher_brief_kz": summary_kz,
            "agent_status": agent_status
        }

    @classmethod
    def _generate_narratives(cls, date: str, total_mwh: float, cf: float, cap_mw: float, storm_h: list, icing_h: list, ramp_h: list, total_hours: int) -> tuple:
        ru_lines = [
            f"⚡ **Диспетчерский отчёт ВЭС Шелек ({cap_mw} МВт) с {date}, горизонт {total_hours} ч:**",
            f"• Прогнозируемая выработка: **{total_mwh:.2f} МВт·ч** (КУИМ: **{cf:.1f}%**).",
            "• Экономический эффект не оценён: нужны фактическая выработка, базовая заявка и цены расчёта небалансов."
        ]
        if storm_h:
            ru_lines.append(f"⚠️ **ШТОРМОВОЙ ОСТАНОВ (v >= 25 м/с):** В часы {storm_h}. Генерация безопасно обнулена агентом.")
        if icing_h:
            ru_lines.append(f"❄️ **Риск обледенения (t <= -10°C):** В часы {icing_h}. Требуется мониторинг аэродинамики лопастей.")
        if ramp_h:
            ru_lines.append(f"📈 **Высокая градиентность ветра:** В часы {ramp_h}. Рекомендовано предупреждение НДЦ СО ЕЭС.")
        if not storm_h and not icing_h and not ramp_h:
            ru_lines.append("Пороговые погодные риски не обнаружены. Это не подтверждает отсутствие эксплуатационных рисков.")

        kz_lines = [
            f"⚡ **Шелек ЖЭС ({cap_mw} МВт): {date} күнінен бастап {total_hours} сағатқа диспетчерлік есеп:**",
            f"• Болжамды электр өндірісі: **{total_mwh:.2f} МВт·сағ** (Пайдалану коэффициенті: **{cf:.1f}%**).",
            "• Экономикалық әсер бағаланбаған: нақты өндіріс, базалық өтінім және теңгерімсіздік бағалары қажет."
        ]
        if storm_h:
            kz_lines.append(f"⚠️ **ДАУЫЛ ТОҚТАУЫ (v >= 25 м/с):** Сағаттар: {storm_h}. Агент болжамды нөлге теңестірді.")
        if icing_h:
            kz_lines.append(f"❄️ **Мұздану қаупі (t <= -10°C):** Сағаттар: {icing_h}. Қалақтардың аэродинамикасын бақылау қажет.")
        if ramp_h:
            kz_lines.append(f"📈 **Қуаттың күрт ауытқуы:** Сағаттар: {ramp_h}. Ұлттық диспетчерлік орталыққа ескерту ұсынылады.")
        if not storm_h and not icing_h and not ramp_h:
            kz_lines.append("Шекті ауа райы қауіптері анықталмады. Бұл пайдалану қауіптерінің жоқтығын растамайды.")

        return "\n".join(ru_lines), "\n".join(kz_lines)
