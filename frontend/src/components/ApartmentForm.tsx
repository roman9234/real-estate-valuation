import { useForm } from 'react-hook-form'
import type { ApartmentRequest, FeaturesMeta } from '../types'

interface Props {
  meta: FeaturesMeta
  loading: boolean
  onSubmit: (data: ApartmentRequest) => void
}

const RENOVATION_LABELS: Record<string, string> = {
  Cosmetic: 'Косметический',
  Designer: 'Дизайнерский',
  'European-style renovation': 'Евроремонт',
  'Without renovation': 'Без ремонта',
}

export default function ApartmentForm({ meta, loading, onSubmit }: Props) {
  const metroValues =
    meta.categorical.find((c) => c.name === 'metro_station')?.values ?? []
  const renovationValues =
    meta.categorical.find((c) => c.name === 'renovation')?.values ?? []

  const num = (name: string) => meta.numeric.find((n) => n.name === name)
  const area = num('area')
  const rooms = num('rooms')
  const floor = num('floor')
  const totalFloors = num('total_floors')
  const metro = num('minutes_to_metro')

  const {
    register,
    handleSubmit,
    watch,
    setValue,
    formState: { errors },
  } = useForm<ApartmentRequest>({
    defaultValues: {
      area: 60,
      rooms: 2,
      floor: 5,
      total_floors: 15,
      minutes_to_metro: 10,
      is_studio: 0,
      metro_station: metroValues[0] ?? '',
      renovation: 'Cosmetic',
    },
  })

  const roomsVal = watch('rooms')
  // Авто-синхронизация студии: 0 комнат => студия.
  if (Number(roomsVal) === 0 && watch('is_studio') !== 1) setValue('is_studio', 1)

  const floorVal = Number(watch('floor'))
  const totalVal = Number(watch('total_floors'))

  return (
    <form className="card" onSubmit={handleSubmit(onSubmit)}>
      <h2 className="card__title">Параметры квартиры</h2>

      <div className="field">
        <label>{area?.label ?? 'Площадь, м²'}</label>
        <input
          type="number"
          step={area?.step ?? 0.1}
          {...register('area', {
            valueAsNumber: true,
            required: 'Обязательное поле',
            min: { value: 10, message: 'Минимум 10 м²' },
            max: { value: 500, message: 'Максимум 500 м²' },
          })}
        />
        {errors.area && <span className="err">{errors.area.message}</span>}
      </div>

      <div className="row">
        <div className="field">
          <label>{rooms?.label ?? 'Комнат'}</label>
          <input
            type="number"
            step={1}
            {...register('rooms', {
              valueAsNumber: true,
              required: true,
              min: { value: 0, message: '0 = студия' },
              max: { value: 15, message: 'Максимум 15' },
            })}
          />
          {errors.rooms && <span className="err">{errors.rooms.message}</span>}
        </div>

        <div className="field">
          <label>Студия</label>
          <select {...register('is_studio', { valueAsNumber: true })}>
            <option value={0}>Нет</option>
            <option value={1}>Да</option>
          </select>
        </div>
      </div>

      <div className="row">
        <div className="field">
          <label>{floor?.label ?? 'Этаж'}</label>
          <input
            type="number"
            step={1}
            {...register('floor', {
              valueAsNumber: true,
              required: true,
              min: { value: 1, message: 'Минимум 1' },
              max: { value: 100, message: 'Максимум 100' },
            })}
          />
          {errors.floor && <span className="err">{errors.floor.message}</span>}
        </div>

        <div className="field">
          <label>{totalFloors?.label ?? 'Этажей в доме'}</label>
          <input
            type="number"
            step={1}
            {...register('total_floors', {
              valueAsNumber: true,
              required: true,
              min: { value: 1, message: 'Минимум 1' },
              max: { value: 100, message: 'Максимум 100' },
            })}
          />
          {errors.total_floors && (
            <span className="err">{errors.total_floors.message}</span>
          )}
        </div>
      </div>

      {/* Бизнес-правило backend: floor <= total_floors */}
      {floorVal > totalVal && (
        <span className="err">Этаж не может быть больше этажности дома</span>
      )}

      <div className="field">
        <label>{metro?.label ?? 'Минут до метро'}</label>
        <input
          type="number"
          step={1}
          {...register('minutes_to_metro', {
            valueAsNumber: true,
            required: true,
            min: { value: 0, message: 'Не может быть отрицательным' },
            max: { value: 120, message: 'Максимум 120' },
          })}
        />
        {errors.minutes_to_metro && (
          <span className="err">{errors.minutes_to_metro.message}</span>
        )}
      </div>

      <div className="field">
        <label>Станция метро</label>
        <select {...register('metro_station', { required: true })}>
          {[...metroValues]
            .sort((a, b) => a.localeCompare(b, 'ru'))
            .map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
        </select>
      </div>

      <div className="field">
        <label>Тип ремонта</label>
        <select {...register('renovation', { required: true })}>
          {renovationValues.map((r) => (
            <option key={r} value={r}>
              {RENOVATION_LABELS[r] ?? r}
            </option>
          ))}
        </select>
      </div>

      <button
        className="btn-primary"
        type="submit"
        disabled={loading || floorVal > totalVal}
      >
        {loading ? 'Расчёт…' : 'Оценить стоимость'}
      </button>
    </form>
  )
}
