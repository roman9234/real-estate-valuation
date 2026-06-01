export default function Loader({ text = 'Загрузка…' }: { text?: string }) {
  return <div className="loader">{text}</div>
}
